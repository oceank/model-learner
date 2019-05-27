# General imports
import numpy as np
from sympy.core import sympify
from learner.lib import *
import re as regex
import random


class Model:
    def __init__(self, terms, ndim):
        self.ndim = ndim
        self.allOptions = ["o" + str(i) for i in range(ndim)]
        self.constant = 0
        self.individualOptions = []
        self.interactions = []
        self.name = ""
        for term in terms:
            if term.isInteraction():
                self.interactions.append(term)
            elif term.isIndividualOption():
                self.individualOptions.append(term)
            elif term.isConstant():
                self.constant = float(term.coefficient)

    def evaluateModel(self, xTest):
        if xTest.shape[1] != self.ndim:
            raise ValueError()

        L = xTest.shape[0]
        r = np.zeros(L)
        f = sympify(self.__str__())
        vars = {}

        for i in range(L):
            for j in range(self.ndim):
                idx = int(regex.findall("\d+$", self.allOptions[j])[0])
                vars[self.allOptions[j]] = xTest[i, idx]
            r[i] = f.subs(vars).evalf()

        return r

    def evaluateModelFast(self, xTest):
        Lo = len(self.individualOptions)
        Li = len(self.interactions)
        A = xTest

        M = np.zeros(self.ndim + Li)

        for i in range(Lo):
            M[self.allOptions.index(self.individualOptions[i].options[0].replace(" ", ""))] = self.individualOptions[
                i].coefficient

        for i in range(Li):
            options = self.interactions[i].options
            coeff = self.interactions[i].coefficient
            M[self.ndim + i] = coeff

            A = np.append(A, A[:, self.allOptions.index(options[0].replace(" ", "")):self.allOptions.index(
                options[0].replace(" ", "")) + 1], axis=1)
            for idx in range(1, len(options)):
                A[:, self.ndim + i] = A[:, self.ndim + i] * A[:, self.allOptions.index(options[idx].replace(" ", ""))]

        r = np.dot(A, M) + self.constant

        return r

    def simplifyModel(self):
        Lo = self.getNumberOfOptions()
        Li = self.getNumberOfInteractions()
        options2remove = []
        for i in range(1, Lo):
            currentOption = self.individualOptions[i]
            for j in range(i):
                if self.individualOptions[j].options[0].replace(" ", "") == currentOption.options[0].replace(" ", ""):
                    self.individualOptions[j].coefficient = self.individualOptions[
                                                                j].coefficient + currentOption.coefficient
                    options2remove.append(i)
                    break

        interactions2remove = []
        for i in range(1, Li):
            currentInteraction = self.interactions[i]
            for j in range(i):
                if len(self.interactions[j].options) == len(currentInteraction.options):
                    equalOptions = 0
                    for k in range(len(self.interactions[j].options)):
                        for l in range(len(currentInteraction.options)):
                            if self.interactions[j].options[k] == currentInteraction.options[l]:
                                equalOptions += 1
                                break
                    if equalOptions == len(self.interactions[j].options):
                        self.interactions[j].coefficient = self.interactions[j].coefficient + currentInteraction.coefficient
                        interactions2remove.append(i)
                        break

        for i in sorted(options2remove, reverse=True):
            self.individualOptions.pop(i)

        for i in sorted(interactions2remove, reverse=True):
            self.interactions.pop(i)

    def getInteractions(self):
        return self.interactions

    def getIndividualOptions(self):
        return self.individualOptions

    def getNumberOfInteractions(self):
        return len(self.interactions)

    def getNumberOfOptions(self):
        return len(self.individualOptions)

    def removeInteraction(self, position):
        if len(self.interactions) >= 1:
            self.interactions.pop(position)

    def removeIndividualOption(self, position):
        if len(self.individualOptions) > 1:  # for a model to be valid, at least one individual option is needed
            self.individualOptions.pop(position)

    def addOption(self, coefficient):
        if len(self.individualOptions) < self.ndim:
            for i in range(self.ndim):
                proposedOption = "o" + str(i)
                shouldBeAdded = True
                for j in range(len(self.individualOptions)):
                    if proposedOption == self.individualOptions[j].options[0].replace(" ", ""):
                        shouldBeAdded = False
                        break
                if shouldBeAdded:
                    self.individualOptions.append(Term(coefficient, [proposedOption]))
                    break
        self.simplifyModel()

    def addInteraction(self, term):
        self.interactions.append(term)
        self.simplifyModel()

    def changeTerm(self, newTerm, position):
        if position < len(self.individualOptions):
            tempTerm = self.individualOptions[position]
            self.individualOptions[position] = newTerm
        else:
            position -= len(self.individualOptions)
            tempTerm = self.interactions[position]
            self.interactions[position] = newTerm
        return tempTerm

    def getTermByPosition(self, position):
        if position < len(self.individualOptions):
            return self.individualOptions[position]
        else:
            return self.interactions[position - len(self.individualOptions)]

    def __str__(self):
        str2 = ""
        Lo = len(self.individualOptions)
        Li = len(self.interactions)
        for i in range(len(self.individualOptions)):
            if i < Lo - 1:
                str2 += str(self.individualOptions[i]) + " + "
            else:
                str2 += str(self.individualOptions[i])

        if Li > 0:
            str2 += " + "
        for i in range(len(self.interactions)):
            if i < Li - 1:
                str2 += str(self.interactions[i]) + " + "
            else:
                str2 += str(self.interactions[i])

        if self.constant != 0:
            str2 += " + " + str(self.constant)
        return str2


class Term:
    def __init__(self, coefficient, options="1"):  # The default value is for the constant term
        self.coefficient = coefficient
        self.options = options

    def __str__(self):
        str2 = str("{0:.2f}".format(self.coefficient)) + " * "
        if len(self.options) > 1:
            for i in range(len(self.options)):
                if i < len(self.options) - 1:
                    str2 += str(self.options[i]) + " * "
                else:
                    str2 += str(self.options[i])
        else:
            str2 += str(self.options[0])
        return str2

    def isConstant(self):
        if len(self.options) == 1 and is_number(self.options[0]):
            return True
        else:
            return False

    def isIndividualOption(self):
        if len(self.options) == 1 and not is_number(self.options[0]):
            return True
        else:
            return False

    def isInteraction(self):
        if len(self.options) == 1:
            return False
        else:
            return True


def genModelTermsfromString(txtModel):
    txtModel = txtModel.replace(" ", "")
    terms = regex.split("[+]", txtModel)
    generatedModel = []
    for i in range(len(terms)):
        term = regex.split("[*]", terms[i])
        if len(term) == 1 and is_number(term[0]):  # this is the constant term
            coeff = float(term[0])
            generatedModel.append(Term(coeff))
        else:
            coeff = 1
            idx = -1
            for index in range(len(term)):
                if is_number(term[index]):
                    coeff = float(term[index])
                    idx = index

            if idx != -1:  # we have a explicit coefficient, i.e., 2*o1 instead of o1
                term.pop(idx)
            generatedModel.append(Term(coeff, term))

    return generatedModel


def genModelfromCoeff(coeff, ndim):
    options = ["o" + str(i) for i in range(ndim)]
    generatedModel = []

    generatedModel.append(Term(coeff[0]))
    for i in range(ndim):
        generatedModel.append(Term(coeff[i+1], [options[i]]))

    for i in range(ndim):
        for j in range(i+1, ndim):
            generatedModel.append(Term(coeff[ndim + 1 + i + j], [options[i], options[j]]))

    return generatedModel


def genModel(individualOptions, interactions, max_coeff):
    given_model = []
    size = individualOptions + interactions
    for i in range(size):
        if i < individualOptions:
            term = Term(random.randint(0, max_coeff), ["o" + str(i)])
            given_model.append(term)
        else:
            option1 = random.randint(0, individualOptions - 1)
            option2 = random.randint(0, individualOptions - 1)
            while option1 == option2:
                option2 = random.randint(0, individualOptions - 1)
            term = Term(random.randint(0, max_coeff), ["o" + str(option1), "o" + str(option2)])
            given_model.append(term)

    return given_model