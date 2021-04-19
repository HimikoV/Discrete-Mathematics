import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pickle as pkl
import ast
import time

towns_with_coordinates = pkl.load(open("city_dictionary.pkl", "rb"))
towns_distances = pkl.load(open("pandas_dict.pkl", "rb"))


def change_dict_to_list(towns_with_coordinates_change):
    return_list = []
    for i in towns_with_coordinates_change:
        return_list.append(i["name"])
    return return_list


def generate_batch(towns_gen, batch_size):
    batch_0 = []
    temporaly_towns = towns_gen
    for i in range(batch_size):
        index = int(np.random.rand() * len(temporaly_towns))
        batch_0.append(temporaly_towns[index])
        temporaly_towns.remove(temporaly_towns[index])

    return batch_0


def distances(batch, starting_point, towns_distances_dist):
    if starting_point in batch:
        batch.remove(starting_point)
    distances_dist = {}
    for i in batch:
        try:
            distances_dist[starting_point + i] = towns_distances_dist[starting_point + i]
        except:
            distances_dist[starting_point + i] = towns_distances_dist[i + starting_point]
    return distances_dist, batch


def desirability(distances_des):
    desirability_des = {}
    for i in distances_des:
        desirability_des[i] = (1 / distances_des[i]) ** 2
    return desirability_des


def random_choose(desirability_ran, tracer, starting_point, distances_ran):
    sum_of_desirabilities = 0
    new_starting_point = None
    flag = 0
    for i in desirability_ran:
        sum_of_desirabilities += desirability_ran[i]
    random_number = np.random.rand() * sum_of_desirabilities

    for i in desirability_ran:
        sum_of_desirabilities -= desirability_ran[i]

        if sum_of_desirabilities < random_number and flag == 0:
            tracer[len(tracer)] = [i[len(starting_point):], distances_ran[i]]
            new_starting_point = i[len(starting_point):]
            flag = 1

    return tracer, new_starting_point


def count_statistics(multi_tracer):
    statistics_pepege = []
    for i in multi_tracer:
        sum_of = 0
        for j in multi_tracer[i]:
            sum_of += multi_tracer[i][j][1]
        statistics_pepege.append(sum_of)
    best_result = min(statistics_pepege)
    indexof = statistics_pepege.index(best_result)
    return multi_tracer[indexof], best_result


def ant_colony(towns_distances_ant, towns_ant, batch_size, number_of_ants=1, random_batch = True, custom_batch = [], custom_starting_point = None):
    batch_0 = None
    starting_point_0 = None
    if random_batch:
        batch_0 = generate_batch(towns_ant, batch_size)
        starting_point_0 = batch_0[int(np.random.rand() * len(batch_0))]
    else:
        batch_0 = custom_batch
        starting_point_0 = custom_starting_point
    batch_0 = str(batch_0)
    multi_tracer = {}
    for j in range(number_of_ants):
        batch = ast.literal_eval(batch_0)
        starting_point = starting_point_0
        tracer = {}
        tracer[0] = [starting_point, 0]
        for i in range(len(batch)):
            distances_1, batch = distances(batch, starting_point, towns_distances_ant)
            desirability_1 = desirability(distances_1)
            tracer, starting_point = random_choose(desirability_1, tracer, starting_point, distances_1)
        multi_tracer[j] = tracer
    return multi_tracer


start = time.time()

towns = change_dict_to_list(towns_with_coordinates)
results = ant_colony(towns_distances, towns, 20, 1000)

best_result_track, best_result_kilometers = count_statistics(results)

end = time.time()
final_time = end-start

for i in best_result_track:
    for j in towns_with_coordinates:
        if best_result_track[i][0] == j["name"]:
            best_result_track[i].append([float(j["lat"]), float(j["long"])])

for i in best_result_track:
    print(best_result_track[i])
print(best_result_kilometers)
print(f"time: {final_time}")
