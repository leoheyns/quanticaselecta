from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer, Aer

from datetime import datetime
import numpy as np
import math
from tensorflow import keras as ks
import tensorflow as tf

import matplotlib.pyplot as plt


class DataGenerator():

    def __init__(self, n_points=250):
        mu_1 = 0.30
        sigma_1 = 0.05
        mu_2 = 0.40
        sigma_2 = 0.04
        self.n_points = n_points
        # self.distribution = np.append(np.random.normal(mu_1, sigma_1, int(self.n_points / 2)),
        #                               np.random.normal(mu_2, sigma_2, int(self.n_points / 2)))
        self.distribution = np.random.normal(mu_1, sigma_1, self.n_points)


class ClipConstraint(ks.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return ks.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return ks.backend.mean(y_true * y_pred)


class VariationalQuantumGAN():

    def __init__(self, n_epochs=100):
        super().__init__()
        # Initialize parameters
        self.backend = Aer.get_backend('aer_simulator_statevector')
        self.backend_sim = Aer.get_backend('qasm_simulator')
        self.latent_space_size = 2
        self.variational_circuit_size = 2

        # Training parameters
        self.n_epochs = n_epochs

        # Parameters initialized as in the paper.
        self.qg_thetas = [2.3, 2.3, 1.0] #, 1.0, 1.5, 0.2]
        # Build circuit with class parameters TODO: Use this to improve efficiency (if needed)
        self.qc = self.build_circuit()

        self.decoding_model = None
        self.build_decoding_model()

        self.data_generator = DataGenerator()
        self.target_dist = self.data_generator.distribution

        self.discriminator = self.build_discriminator()

        # Build gan stack for fitting the generator
        self.gan_stack = ks.Sequential()
        self.gan_stack.add(self.decoding_model)
        self.gan_stack.add(self.discriminator)
        # TODO: Check learning rate
        opt = ks.optimizers.RMSprop(lr=0.0005)
        self.gan_stack.compile(loss=wasserstein_loss, optimizer=opt)

    def build_circuit(self, measurement=True):
        qc = QuantumCircuit(self.variational_circuit_size, self.variational_circuit_size)

        # Quantum encoding.
        # z ~ U[-1, 1]
        z = np.random.uniform(-1, 1, self.latent_space_size)

        for i in range(self.latent_space_size):
            qc.rx(1 / math.sin(z[i]), i + (self.variational_circuit_size - self.latent_space_size))
            qc.rz(1 / math.cos(z[i]), i + (self.variational_circuit_size - self.latent_space_size))

        qc.barrier()

        # Variational Circuit
        # TODO: Scale dynamically with circuit size?
        qc.ry(self.qg_thetas[0], 0)
        qc.ry(self.qg_thetas[1], 1)
        qc.rxx(self.qg_thetas[2], 0, 1)
        qc.barrier()

        # qc.ry(self.qg_thetas[3], 0)
        # qc.ry(self.qg_thetas[4], 1)
        # qc.rxx(self.qg_thetas[5], 0, 1)
        # qc.barrier()

        # Measurement decoding
        if measurement:
            qc.measure(0, 0)
            qc.measure(1, 1)
        else:
            qc.save_statevector()

        return qc

    # Build circuit with params as arguments, used for parameter shift gradient calculation
    def build_circuit_with_params(self, params, measurement=False):
        qc = QuantumCircuit(self.variational_circuit_size, self.variational_circuit_size)

        # Quantum encoding.
        # z ~ U[-1, 1]
        z = np.random.uniform(-1, 1, self.latent_space_size)

        for i in range(self.latent_space_size):
            qc.rx(1 / math.sin(z[i]), i + (self.variational_circuit_size - self.latent_space_size))
            qc.rz(1 / math.cos(z[i]), i + (self.variational_circuit_size - self.latent_space_size))

        qc.barrier()

        # Variational Circuit
        # TODO: Scale dynamically with circuit size?
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.rxx(params[2], 0, 1)
        qc.barrier()

        # qc.ry(params[3], 0)
        # qc.ry(params[4], 1)
        # qc.rxx(params[5], 0, 1)
        # qc.barrier()

        # Measurement decoding
        if measurement:
            qc.measure(0, 0)
            qc.measure(1, 1)
        else:
            qc.save_statevector()

        return qc

    def build_discriminator(self):
        # define the constraint
        const = ClipConstraint(0.01)

        model = ks.Sequential()
        model.add(ks.layers.Dense(50, activation='elu', input_shape=(1,), kernel_constraint=const))
        #         model.add(ks.layers.Dense(100, activation='relu', kernel_constraint=const))
        model.add(ks.layers.Dense(50, activation='elu', kernel_constraint=const))
        model.add(ks.layers.Dense(1, activation='linear', kernel_constraint=const))

        # TODO: Investigate loss
        opt = ks.optimizers.RMSprop(lr=0.005)
        model.compile(loss=wasserstein_loss, optimizer=opt)
        return model

    def build_decoding_model(self):
        x = ks.layers.Input(shape=(self.variational_circuit_size,), name="input")
        decoding_layer = ks.layers.Dense(1, activation="sigmoid", name="decoding_layer")
        y = decoding_layer(x)

        self.decoding_model = ks.Model(inputs=x, outputs=y)
        self.decoding_model.summary()

    # Generate measurement with the class
    def generate_measurement(self, qc=None, statevector=False):
        # Somehow statevector is slower: TODO figure out. 25 seconds sv vs 15 seconds 100 shot sims.
        if not statevector:
            # TODO: change to use a class qc
            if qc is None:
                qc = self.build_circuit()
            sim_shots = 10000

            job = self.backend_sim.run(qc, shots=sim_shots)
            result_sim = job.result()

            counts = result_sim.get_counts(qc)
            #         print(counts)

            classical_latent_space = np.zeros(self.variational_circuit_size)
            for count in counts.keys():
                v_n = 0
                for bit in count:
                    if bit == '1':
                        classical_latent_space[v_n] += counts[count]
                    v_n += 1

            classical_latent_space = np.stack([classical_latent_space / sim_shots])

            return classical_latent_space

        else:
            if qc is None:
                qc_no_measure = self.build_circuit(measurement=False)
                # qc_no_measure.save_statevector()
            else:
                qc_no_measure = qc

            # Transpile for simulator
            simulator = Aer.get_backend('aer_simulator')
            circ = transpile(qc_no_measure, simulator)

            # Run and get statevector
            result = simulator.run(circ).result()
            statevector = result.get_statevector(circ)

            collapsed_sv = np.abs(np.stack(statevector)) ** 2

            classical_latent_space = np.zeros(self.variational_circuit_size)
            for i in range(len(collapsed_sv)):
                # Counts are fixed. CHANGE 02b TO NEW LATENT SPACE SIZE IF CHANGED
                current_count = "{0:02b}".format(i)
                for bit in range(len(current_count)):
                    if current_count[bit] == "1":
                        classical_latent_space[bit] += collapsed_sv[i]

            classical_latent_space = classical_latent_space.reshape(1, self.variational_circuit_size)

            return classical_latent_space

    def generate_prediction(self, qc=None):
        circuit_measurement = self.generate_measurement(qc=qc, statevector=True)
        return self.decoding_model.predict(circuit_measurement)

    def generate_fake_samples(self, n):
        # print("Generating fake samples for training")
        predictions = []
        for i in range(n):
            predictions.append(self.generate_prediction())

        predictions = np.stack(predictions)
        predictions = predictions.reshape((n, 1))

        return predictions, np.ones((n, 1))

    def generate_fake_samples(self, n, qc=None):
        # print("Generating fake samples for training")
        predictions = []
        for i in range(n):
            predictions.append(self.generate_prediction(qc=qc))

        predictions = np.stack(predictions)
        predictions = predictions.reshape((n, 1))

        return predictions, np.ones((n, 1))

    def get_fixed_critic_belief(self, x):
        if x[0] < 0.25:
            return -x[0] * 0.02
        else:
            return 0.006667 * x[0] - 0.006667

    def train(self):
        self.discriminator.summary()

        # TODO: Resample real distribution continuously instead of using fixed amount of points
        X_real = np.stack(self.target_dist).reshape(len(self.target_dist), 1)
        y_real = -np.ones((len(self.target_dist), 1))

        n_critic = 5
        gen_batch_size = 32
        disc_batch_size = gen_batch_size * 2

        for i in range(self.n_epochs):
            X_fake_sampled, y_fake_sampled = self.generate_fake_samples(int(disc_batch_size / 2))
            # Train discriminator
            # TODO: Perform basic improvements on discriminator

            for _ in range(n_critic):
                # Select read and fake data_points, we assume N_fake == N_real (Amount fake datapoints is equal to amount of real datapoints)
                idx = np.random.choice(np.arange(len(X_real)), int(disc_batch_size / 2))

                # Train on real
                X_real_sampled, y_real_sampled = np.take(X_real, idx), np.take(y_real, idx)

                # # preds = self.get_fixed_critic_belief_on_batch
                # self.discriminator.train_on_batch(X_real_sampled, y_real_sampled)
                #
                # # Train on fake
                # self.discriminator.train_on_batch(X_fake_sampled, y_fake_sampled)

            print("Iteration", i)  # , "Real accuracy", acc_real, "Fake accuracy", acc_fake)

            # Train generator stack

            # Calculate Generator Cost
            disc_pred = 0
            for j in range(len(X_fake_sampled)):
                disc_pred += self.get_fixed_critic_belief(X_fake_sampled[j]) / disc_batch_size

            print("Gen cost:", disc_pred)

            self.discriminator.trainable = False

            # Update Quantum and Classical part of generator with parameter shift
            ### BULK METHOD ###
            # before = datetime.now()
            # print("Start bulk method")
            # non_shift_measurements = []
            # non_shift_cost = 0
            # gradients = [0] * len(self.qg_thetas)
            #
            # non_shift_qc = self.build_circuit_with_params(self.qg_thetas)
            #
            # neg_shift_qcs = [None] * len(self.qg_thetas)
            # pos_shift_qcs = [None] * len(self.qg_thetas)
            #
            # for k in range(len(self.qg_thetas)):
            #     neg_shift_params = self.qg_thetas.copy()
            #     neg_shift_params[k] -= math.pi / 2
            #     neg_shift_qc = self.build_circuit_with_params(neg_shift_params)
            #     neg_shift_qcs[k] = neg_shift_qc
            #
            #     pos_shift_params = self.qg_thetas.copy()
            #     pos_shift_params[k] += math.pi / 2
            #     pos_shift_qc = self.build_circuit_with_params(pos_shift_params)
            #     pos_shift_qcs[k] = pos_shift_qc
            #
            # for j in range(gen_batch_size):
            #     non_shift_measurement = self.generate_measurement(qc=non_shift_qc, statevector=False)
            #     non_shift_sample = self.decoding_model.predict(non_shift_measurement)
            #     non_shift_cost += self.get_fixed_critic_belief([non_shift_sample])
            #     non_shift_measurements.append(non_shift_measurement)
            #
            #     for k in range(len(neg_shift_qcs)):
            #         neg_shift_sample = self.generate_prediction(qc=neg_shift_qcs[k])
            #         neg_shift_cost = self.get_fixed_critic_belief([neg_shift_sample])
            #
            #         pos_shift_sample = self.generate_prediction(qc=pos_shift_qcs[k])
            #         pos_shift_cost = self.get_fixed_critic_belief([pos_shift_sample])
            #
            #         gradients[k] += pos_shift_cost - neg_shift_cost
            #
            # non_shift_cost = non_shift_cost / gen_batch_size
            # non_shift_measurements = np.reshape(np.stack(non_shift_measurements), newshape=(gen_batch_size, self.variational_circuit_size))
            #
            # for j in range(len(gradients)):
            #     self.qg_thetas[j] += gradients[j] * non_shift_cost
            #
            # self.gan_stack.train_on_batch(np.stack(non_shift_measurements), -np.ones(gen_batch_size).astype(float))

            # after = datetime.now()
            # print("Finish bulk method, took", str(after - before))
            ### SINGLE BY SINGLE METHOD ###
            before = datetime.now()
            print("Start single method")
            for j in range(gen_batch_size):
                # before = datetime.now()
                # print("Train decoder model")
                non_shift_qc = self.build_circuit_with_params(self.qg_thetas)
                non_shift_measurement = self.generate_measurement(qc=non_shift_qc, statevector=False)
                #                 print(non_shift_measurement)
                non_shift_sample = self.decoding_model.predict(non_shift_measurement)
                non_shift_cost = self.get_fixed_critic_belief([non_shift_sample])

                # after = datetime.now()
                # print("Done generating non shifted qcs, took", str(after - before))

                # before = datetime.now()
                # print("Train decoder model")
                self.gan_stack.train_on_batch(np.stack(non_shift_measurement), np.stack([-1]).astype(float))
                # after = datetime.now()
                # print("Done: Train decoder model, took", str(after - before))
                #
                # before = datetime.now()
                # print("Param shift")
                for k in range(len(self.qg_thetas)):
                    # before = datetime.now()
                    # print("Neg shifts")
                    neg_shift_params = self.qg_thetas.copy()
                    neg_shift_params[k] -= math.pi / 2
                    neg_shift_qc = self.build_circuit_with_params(neg_shift_params)

                    neg_shift_sample = self.generate_prediction(qc=neg_shift_qc)
                    neg_shift_cost = self.get_fixed_critic_belief([neg_shift_sample])

                    # after = datetime.now()
                    # print("Done neg shifts, took", str(after - before))

                    # before = datetime.now()
                    # print("Pos shifts")
                    pos_shift_params = self.qg_thetas.copy()
                    pos_shift_params[k] += math.pi / 2
                    pos_shift_qc = self.build_circuit_with_params(pos_shift_params)

                    pos_shift_sample = self.generate_prediction(qc=pos_shift_qc)
                    pos_shift_cost = self.get_fixed_critic_belief([pos_shift_sample])

                    # after = datetime.now()
                    # print("Done pos shifts, took", str(after - before))

                    gradient = neg_shift_cost - pos_shift_cost

                    self.qg_thetas[k] += gradient * non_shift_cost
                # after = datetime.now()
                # print("Done param shift, took", str(after - before))
            after = datetime.now()
            print("Finish single method, took", str(after - before), "new params", self.qg_thetas)

            self.discriminator.trainable = True

            if i % 5 == 0:
                self.create_analysis_plots(iteration=i)

    def create_analysis_plots(self, iteration):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # TODO: Combine these into 1 method and for gods sake make it efficient
        self.generate_critic_belief(plot=True, iteration=iteration, subplot=axs)
        self.generate_generator_range(plot=True, iteration=iteration, subplot=axs)
        self.generate_qc_range(plot=True, iteration=iteration, subplot=axs)
        self.compare_model_to_real(iteration=iteration, subplot=axs)

        fig.show()

    def generate_critic_belief(self, plot=False, iteration=-1, subplot=None):
        values = np.arange(0, 1, 0.01)
        beliefs = []
        for i in range(len(values)):
            beliefs.append(self.get_fixed_critic_belief([values[i]]))

        beliefs = np.reshape(beliefs, 100)

        if plot:
            subplot[0, 0].plot(np.arange(0, 1, 0.01), beliefs, label="belief")
            subplot[0, 0].set_title("Belief at iteration " + str(iteration))
            #             plt.ylim(-1, 1)
            # plt.show()

        return beliefs

    def generate_qc_range(self, plot=True, iteration=-1, subplot=None):
        values = []
        colors = []
        for i in range(100):
            measurement = self.generate_measurement(statevector=True)
            values.append(measurement)

            generated_value = self.decoding_model.predict(measurement)
            # The brighter the color the higher the decoded value is
            colors.append(generated_value)

        values = np.stack(values)

        x = values[:, :, 0]
        y = values[:, :, 1]

        if plot:
            sc = subplot[0, 1].scatter(x, y, c=colors)
            subplot[0, 1].scatter(np.mean(x), np.mean(y), label="average_point", color='red')
            subplot[0, 1].set_title("Values generated by Variational QC at iteration " + str(iteration))
            subplot[0, 1].set_xlim(0, 1)
            subplot[0, 1].set_ylim(0, 1)
            subplot[0, 1].legend()

            plt.colorbar(sc, ax=subplot[0, 1])
            # plt.show()

        return values

    def generate_generator_range(self, plot=True, iteration=-1, subplot=None):
        x1s = np.arange(0, 1, 0.05)
        x2s = np.arange(0, 1, 0.05)
        generated_values = []
        for x1 in x1s:
            for x2 in x2s:
                generated_values.append(self.decoding_model.predict([[x1, x2]]))

        bins = np.linspace(0, 1, 100)

        generated_values = np.reshape(np.stack(generated_values), newshape=(len(generated_values)))

        if plot:
            subplot[1, 0].hist(generated_values, bins, density=True)
            subplot[1, 0].set_title("Possible values generated by decoder at iteration " + str(iteration))
            # plt.show()

        return generated_values

    def compare_model_to_real(self, iteration=-1, subplot=None):
        bins = np.linspace(0, 1, 50)

        plt.hist(self.target_dist, bins, density=True, label="True")
        predictions = []
        self.printProgressBar(iteration=0, total=10)
        j = 0
        for i in range(self.data_generator.n_points):
            predictions.append(self.generate_prediction())
            if i % (self.data_generator.n_points / 10) == 0:
                j += 1
                self.printProgressBar(iteration=j, total=10)
        #                 print(str(i/self.data_generator.n_points*100) + "%")

        predictions = np.stack(predictions)
        predictions = predictions.reshape((self.data_generator.n_points,))

        subplot[1, 1].hist(predictions, bins, density=True, label="False")

        # subplot[1, 1].plot(np.arange(0, 1, 0.01), self.generate_critic_belief(), label="Critic belief")

        subplot[1, 1].set_title("Real vs Fake sampled at iteration " + str(iteration))
        subplot[1, 1].legend()
        # plt.show()

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()


vqgan = VariationalQuantumGAN()
vqgan.train()
# vqgan.generate_qc_range(plot=True)
# vqgan.generate_generator_range()
# vqgan.create_analysis_plots(0)
