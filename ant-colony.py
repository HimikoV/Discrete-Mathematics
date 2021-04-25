import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import ast
import time


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


def desirability(distances_des, power=2):
    desirability_des = {}
    for i in distances_des:
        desirability_des[i] = (1 / distances_des[i]) ** power
    return desirability_des


def random_choose(desirability_ran, tracer, starting_point, distances_ran, ):
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
    all_statistics = {}
    for x in multi_tracer:
        statistics_pepege = []
        for k in multi_tracer[x][0]:
            sum_of = 0
            for l in multi_tracer[x][0][k]:
                sum_of += multi_tracer[x][0][k][l][1]
            statistics_pepege.append(sum_of)
        all_statistics[x]=statistics_pepege
    list_of_best_result = []
    list_of_indexes = []
    for x in multi_tracer:
        list_of_best_result.append(min(all_statistics[x]))
        list_of_indexes.append(all_statistics[x].index(list_of_best_result[x]))
    best_result = min(list_of_best_result)
    indexof = list_of_best_result.index(best_result)
    return multi_tracer[indexof][0][list_of_indexes[indexof]], best_result, indexof


def ant_colony(towns_distances_ant, towns_ant, batch_size, number_of_ants=1, power=[2], random_batch=True,
               custom_batch=[], custom_starting_point=None):
    batch_0 = None
    starting_point_0 = None
    if random_batch:
        batch_0 = generate_batch(towns_ant, batch_size)
        starting_point_0 = batch_0[int(np.random.rand() * len(batch_0))]
    else:
        batch_0 = custom_batch
        starting_point_0 = custom_starting_point
    batch_0 = str(batch_0)
    all_simulations = {}
    for x in range(len(power)):
        multi_tracer = {}
        for j in range(number_of_ants):
            batch = ast.literal_eval(batch_0)
            starting_point = starting_point_0
            tracer = {}
            tracer[0] = [starting_point, 0]
            for i in range(len(batch)):
                distances_1, batch = distances(batch, starting_point, towns_distances_ant)
                desirability_1 = desirability(distances_1, power[x])
                tracer, starting_point = random_choose(desirability_1, tracer, starting_point, distances_1)
            multi_tracer[j] = tracer
        all_simulations[x] = [multi_tracer, power[x]]
    return all_simulations


# data loading
towns_with_coordinates = pkl.load(open("city_dictionary.pkl", "rb"))
towns_distances = pkl.load(open("pandas_dict.pkl", "rb"))
power=[1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0,
       5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0,
       9.2, 9.4, 9.6, 9.8, 10.0, ]
"""symulation is working here"""
start = time.time()
towns = change_dict_to_list(towns_with_coordinates)
# you have to specify number of towns, number of iteration,
# list of power wages for each simulation with therefore specified number of iterations
results = ant_colony(towns_distances, towns, 20, 10000, power)
best_result_track, best_result_kilometers, power_index = count_statistics(results)
end = time.time()
final_time = end - start

# choosing best result from a dictionary with all results across every simulation
for i in best_result_track:
    for j in towns_with_coordinates:
        if best_result_track[i][0] == j["name"]:
            best_result_track[i].append([float(j["lat"]), float(j["long"])])

points_x = []
points_y = []
for i in best_result_track:
    print(best_result_track[i])
    points_x.append(best_result_track[i][2][0])
    points_y.append(best_result_track[i][2][1])
print(f"best distance: {best_result_kilometers}")
print(f"time: {final_time} seconds")
print(f"best power from all simulations: {power[power_index]}")

plt.scatter(points_x[:1], points_y[:1], color=["black"])
plt.scatter(points_x[1:], points_y[1:], color=["green"])
plt.plot(points_x, points_y, color="black")
plt.show()
