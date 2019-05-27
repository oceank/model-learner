def is_number(str):
    try:
        # for int, long, float and complex
        complex(str)
    except ValueError:
        return False

    return True


def get_json(pareto_power, pareto_speed):

    data = {}
    data['configurations'] = []
    for i in range(len(pareto_power)):
        data['configurations'].append({
            'config_id': i + 1,
            'power_load': pareto_power[i]/3600*1000,
            'power_load_w': pareto_power[i],
            'speed': pareto_speed[i]
        })

    return data
