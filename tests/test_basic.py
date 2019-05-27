# -*- coding: utf-8 -*-
import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_learner(self):
        assert True


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(BasicTestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    unittest.main()
