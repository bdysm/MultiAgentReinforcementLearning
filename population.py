import multiprocessing
from sklearn.metrics import mean_squared_error

import sswn_weights
from response import Response
from sswn_weights import Weights


class Population(multiprocessing.Process):

    def __init__(self, structure, size):
        super().__init__()
        self.size = size
        self.genotyp = self.genotyp_init(structure)

    def genotyp_init(self, structure):
        genotyp = []
        for i in range(self.size):
            genotyp.append(Gen(structure))
        return genotyp

    def compute(self, structure, u_input, y_target):

        for i in range(self.size):
            resp = Response(u_input, self.genotyp[i].weight, structure)
            self.genotyp[i].mse = mean_squared_error(resp.output[:, -6:], y_target[:, -6:])

    def sort(self):
        self.genotyp = sorted(self.genotyp, key=lambda genotyp: genotyp.mse)

    def get_best_mse(self):
        self.sort()
        return self.genotyp[0].mse

    def get_best_genotyp(self):
        self.sort()
        return self.genotyp[0]

    def get_best_gens(self, best_size, structure):
        if best_size > self.size:
            return self
        else:
            new_pop = Population(structure, best_size)
            for i in range(best_size):
                new_pop.genotyp[i] = self.genotyp[i]
            return new_pop

    def update(self, best_size, random_size, anneal_size, structure):
        self.sort()

        if self.size > (best_size+anneal_size):
            for i in range(anneal_size):
                # self.genotyp[best_size + random_size + i].g_anneal(anneal_temperature)
                self.genotyp[best_size + random_size + i] = self.gen_anneal(structure)

        if best_size < self.size:
            for i in range(random_size):
                self.genotyp[best_size + i] = Gen(structure)

    def concatenate(self, second_population):
        self.size = self.size + second_population.size
        for i in range(second_population.size):
            self.genotyp.append(second_population.genotyp[i])

    def gen_anneal(self, structure):
        gen = Gen(structure)
        gen.weight.values[:] = self.genotyp[0].weight.values[:]
        gen.g_anneal()
        return gen

    def anneal(self, structure, row, col):
        self.genotyp[0].weight.values[row, col] = sswn_weights.Weights.anneal_single(self.genotyp[0].weight)


class Gen:

    def __init__(self, structure):
        self.weight = Weights(structure)
        self.mse = None

    def g_anneal(self):
        self.weight.anneal_all()
        self.mse = None

    def g_anneal_single(self):
        self.weight.anneal_single()
        self.mse = None