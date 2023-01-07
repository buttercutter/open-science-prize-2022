import torch.optim as optim
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import decimal
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts

# Importing standard Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, IBMQ, execute, transpile, assemble, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.providers.backend import Backend

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector

from qiskit.opflow import Zero, One, X, Y, Z, I, H, CX, CircuitStateFn, StateFn, PauliExpectation, CircuitSampler, AerPauliExpectation, MatrixExpectation
from qiskit.opflow.expectations import ExpectationFactory

from typing import Dict, List, Optional, Union, Tuple
import sys
from contextlib import closing
from io import StringIO

import gym
from gym import spaces
from gym.utils import seeding

debug = False

def get_default_gates_qiskit(qubits):
    gates = []
    for qubit in range(qubits):
        next_qubit = (qubit + 1) % qubits
        cX = QuantumCircuit(qubits)
        cX.x(qubit)
        cY = QuantumCircuit(qubits)
        cY.y(qubit)
        cZ = QuantumCircuit(qubits)
        cZ.z(qubit)
        cH = QuantumCircuit(qubits)
        cH.h(qubit)
        cRZ = QuantumCircuit(qubits)
        cRZ.rz(np.pi / 4., qubit)
        cCNOT = QuantumCircuit(qubits)
        cCNOT.cnot(qubit, next_qubit)
        gates += [cX,cY,cZ,cH,cRZ,cCNOT]
    return gates

def get_default_observables_qiskit(qubits):
    observables = []
    states = [X,Y,Z]
    for qubit in range(qubits):
        for state in states:
            i = qubits - (qubit +1)
            j = qubit
            observables.append((I ^ i) ^ state ^ (I ^ j))
    return observables

def get_bell_state() -> np.ndarray:
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target

def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target

def state_tomo(result, st_qcs, target_state):
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    fid = state_fidelity(rho_fit, target_state)
    return fid

class QuantumSearchEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(
        self,
        simulator: Backend,
        estimator,
        custom_vqe,
        target,
        target_energy,
        qubits: List,
        state_observables: List,
        action_gates: List,
        fidelity_threshold: float,
        reward_penalty: float,
        max_timesteps: int,
        error_observables: Optional[float] = None,
        error_gates: Optional[float] = None
    ):
        super(QuantumSearchEnv, self).__init__()

        # set parameters
        self.target = target
        self.target_energy = target_energy
        self.qubits = qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates

        # set environment
        self.target_density = target.to_matrix() * np.conj(target.to_matrix()).T
        self.simulator = simulator
        self.estimator = estimator
        self.custom_vqe = custom_vqe
        
        #if debug:
        print("init simulator", self.simulator)

        # set spaces
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.seed()

    def __str__(self):
        desc = 'QuantumSearch-v0('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target.to_matrix())
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.circuit_gates = []
        return self._get_obs()
    
    def rel_err(self, target, measured):
        return abs((target - measured) / target)

    def _get_circuit(self):
        circuit = QuantumCircuit(self.qubits)
        target_qubits = []
        for x in range(self.qubits):
            target_qubits.append(x)
        for qubit in range(self.qubits):
            circuit.i(qubit) 
        for gate in self.circuit_gates:
            circuit.append(gate, target_qubits)
        return circuit

    def _get_obs(self):
        circuit = self._get_circuit()
        obs = []
        for observable in self.state_observables:
            exp_converter = ExpectationFactory.build(observable, self.simulator)
            measurable_expression =  ~StateFn(observable) @ StateFn(circuit)
            expect_op = exp_converter.convert(measurable_expression)
            sampled_op = CircuitSampler(self.simulator).convert(expect_op)
            expectation_value = sampled_op.eval().real        
            obs.append(expectation_value)
        return np.array(obs).real

    def _get_fidelity(self):
        circuit = self._get_circuit()
        self.custom_vqe._circuit = circuit
        result = self.custom_vqe.compute_minimum_eigenvalue(self.target)
        # Compute the relative error between the expected ground state energy and the VQE's output
        fidelity = self.rel_err(self.target_energy, result.eigenvalue)
        #if debug and fidelity > 0.6:
        print(f'{fidelity:.3f}')
        display(circuit.decompose().draw())
        return fidelity

    def step(self, action):

        # update circuit
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)
        
        # compute observation
        observation = self._get_obs()

        # compute fidelity
        fidelity = self._get_fidelity()

        # compute reward
        if fidelity < self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        # check if terminal
        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)
        print("observation:", observation, " fidelity:", fidelity, " > fid treshold:", self.fidelity_threshold, " reward:", reward, " gates:", len(self.circuit_gates), " action:", action)
        # return info
        info = {'fidelity': fidelity, 'circuit': self._get_circuit()}

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_circuit().__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
    
class BasicNQubitEnv(QuantumSearchEnv):
    def __init__(self,
                 simulator: Backend,
                 estimator,
                 custom_vqe,
                 target,
                 target_energy,
                 fidelity_threshold: float = 0.0001,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 16,
                 action_gates: List = None,
                 problem_size: int = 0):
        n_qubits = int(np.log2(len(target.to_matrix())))
        if (problem_size > 0):
            n_qubits = problem_size
        qubits = n_qubits
        state_observables = get_default_observables_qiskit(qubits)
        if action_gates is None:
            action_gates = get_default_gates_qiskit(qubits)
        super(BasicNQubitEnv,
              self).__init__(simulator, estimator, custom_vqe, target, target_energy, qubits, state_observables, action_gates,
                             fidelity_threshold, reward_penalty, max_timesteps)
        
def learnEnvironmentA2C(env, train_cycle_max):
    # Parameters
    gamma = 0.99
    learning_rate = 0.0001
    policy_kwargs = dict(optimizer_class=optim.Adam)

    # Agent
    model = A2C("MlpPolicy",
                    env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=train_cycle_max)
    return model

def learnEnvironmentDQN(env, train_cycle_max):
    # Parameters
    gamma = 0.99
    learning_rate = 0.0001
    policy_kwargs = dict(optimizer_class=optim.Adam)

    # Agent
    model = DQN("MlpPolicy",
                    env,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=train_cycle_max)
    return model

def saveModel(model, model_name):
    model.save(model_name)

def loadModel(typeAlgo, model_name):
    model = None
    if typeAlgo == "A2C": 
        model = A2C.load(model_name)
    elif typeAlgo == "DQN": 
        model = DQN.load(model_name)
    return model
    
def createRLOptimizationCircuit(Trotter_RL_Operations, model_name, train_method, target_matrix, target_energy, train_cycle_max, backend, problem_size, estimator, custom_vqe):
    # Parameters 
    fidelity_threshold = 0.0001
    reward_penalty = 0.01
    max_timesteps = train_cycle_max
    target = target_matrix
    # Environment
    env = BasicNQubitEnv(simulator=backend,
                   estimator = estimator,
                   custom_vqe = custom_vqe,
                   target = target,
                   target_energy = target_energy,
                   fidelity_threshold=fidelity_threshold,
                   reward_penalty=reward_penalty,
                   max_timesteps=max_timesteps,
                   action_gates=Trotter_RL_Operations,
                   problem_size=problem_size)
    
    model = None
    if train_method == "TRAIN":
        model = learnEnvironmentDQN(env, train_cycle_max)
        saveModel(model, model_name)
    if train_method == "LOAD":
        model = loadModel("DQN", model_name)
    if train_method == "TEST":
        model = learnEnvironmentDQN(env, train_cycle_max)
        
    print("DONE")
    circuit_generated = None
    best_fidelity = 0
    for jj in range(1):
        obs = env.reset()
        for i in range(1):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if info['fidelity'] < best_fidelity:
                circuit_generated = info['circuit']
                best_fidelity = info['fidelity']

    print("generated, best_fidelity \n", circuit_generated.decompose(), best_fidelity)
                
    return circuit_generated

def errorMitigation(qr, q_regs, backend, error_mit):
    if error_mit == 'complete':
        cal_circuits, state_labels = complete_meas_cal(qr=qr,qubit_list=q_regs, 
                                                   circlabel='measurement_calibration')#name
    elif error_mit == 'tensored':
        cal_circuits, state_labels = tensored_meas_cal(qr=qr,mit_pattern=[q_regs], 
                                                   circlabel='measurement_calibration')#name
    loadjob = False
    
    if loadjob == False: 
        cal_job = execute(cal_circuits,
                 backend=backend,
                 shots=15000,
                 optimization_level=0)
        job_monitor(cal_job)
    else : 
        cal_job = backend.retrieve_job('62e65d5422b5f5ebd55679d9')
        
    cal_results = cal_job.result()
    if error_mit == 'complete':
        meas_fitter = CompleteMeasFitter(cal_results, state_labels)
    elif error_mit == 'tensored':
        meas_fitter = TensoredMeasFitter(cal_results, state_labels)
    meas_fitter.plot_calibration()
    return meas_fitter

 # Build a subcircuit for XX(t) two-qubit gate
def compute_XX_gate(t, first, second, size):
    XX_qr = QuantumRegister(size)
    XX_qc = QuantumCircuit(XX_qr, name='XX' + str(first))

    XX_qc.ry(np.pi/2,[first, second])
    XX_qc.cnot(first, second)
    XX_qc.rz(2 * t, second)
    XX_qc.cnot(first, second)
    XX_qc.ry(-np.pi/2,[first, second])

    # Convert custom quantum circuit into a gate
    XX = XX_qc
    
    return XX
# Build a subcircuit for YY(t) two-qubit gate
def compute_YY_gate(t, first, second, size):
    # FILL YOUR CODE IN HERE
    YY_qr = QuantumRegister(size)
    YY_qc = QuantumCircuit(YY_qr, name='YY' + str(first))

    YY_qc.rx(np.pi/2,[first, second])
    YY_qc.cnot(first, second)
    YY_qc.rz(2 * t, second)
    YY_qc.cnot(first, second)
    YY_qc.rx(-np.pi/2,[first, second])

    # Convert custom quantum circuit into a gate
    YY = YY_qc
    
    return YY

# Build a subcircuit for ZZ(t) two-qubit gate
def compute_ZZ_gate(t, first, second, size):
    # FILL YOUR CODE IN HERE
    ZZ_qr = QuantumRegister(size)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ' + str(first))

    ZZ_qc.cnot(first, second)
    ZZ_qc.rz(2 * t, second)
    ZZ_qc.cnot(first, second)

    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc

    return ZZ

def compute_RL_circuit(Trotter_RL_Operations, model_name, train_method, target_matrix, target_energy, train_cycle_max, backend, problem_size, estimator, custom_vqe):
    circuit = createRLOptimizationCircuit(Trotter_RL_Operations, model_name, train_method, target_matrix, target_energy, train_cycle_max, backend, problem_size, estimator, custom_vqe)
    display(circuit.draw())
    return circuit

def create_circuit(model_name, train_method, target_matrix, target_energy, profile, train_cycle_max, q_regs, trotter_steps, target_parameter, backend, problem_size, estimator, custom_vqe):
    num_qubits = int(np.log2(len(target_matrix.to_matrix())))
    
    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')
    RL_qc = None
    
    Trotter_RL_Operations = []
    if profile == "TROTTER" or profile == "ALL":
        for trotter_position in range(num_qubits-1):
            Trotter_RL_Operations += [compute_XX_gate(target_parameter,trotter_position,trotter_position+1, num_qubits),
                                     compute_YY_gate(target_parameter,trotter_position,trotter_position+1,  num_qubits),
                                     compute_ZZ_gate(target_parameter,trotter_position,trotter_position+1,  num_qubits)]
    elif profile == "GATES" or profile == "ALL":
        Trotter_RL_Operations += get_default_gates_qiskit(num_qubits)
    
    RL_qc = compute_RL_circuit(Trotter_RL_Operations, model_name, train_method, target_matrix, target_energy, train_cycle_max, backend,problem_size, estimator, custom_vqe)
        
    if RL_qc is not None:
        target_qubits = []
        for x in range(num_qubits):
            target_qubits.append(x)
        Trot_qc.append(RL_qc, target_qubits)

    return Trot_qc

def runRLOptimizedCircuit(model_name, train_method, target_matrix, target_energy, profile, train_cycle_max, problem_size, backend, estimator, custom_vqe):
    # Setup experiment parameters

    # The final time of the state evolution
    target_time = np.pi  # DO NOT MODIFY

    # Number of trotter steps
    trotter_steps = 8  ### CAN BE >= 4

    # Qiskit parameter is changed to a normal parameter
    target_parameter = target_time/trotter_steps

    # Select which qubits to use for the simulation
    q_regs = []
    for x in range(int(np.log2(len(target_matrix.to_matrix())))):
        q_regs.append(x)

    # Convert custom quantum circuit into a gate
    Trot_gate = create_circuit(model_name, train_method, target_matrix, target_energy, profile, train_cycle_max, q_regs, trotter_steps, target_parameter, backend, problem_size, estimator, custom_vqe).to_instruction()

    # Initialize quantum circuit for 3 qubits
    qr = QuantumRegister(problem_size)
    qc = QuantumCircuit(qr)

    if profile == "TROTTER":
        # Prepare initial state (remember we are only evolving 3 of the 7 qubits on manila qubits (q_5, q_3, q_1) corresponding to the state |110>)
        qc.x([q_regs[2], q_regs[1]])  # For example this could be (q_regs=[2, 1, 0] which corresponds to => |110>)

        # Simulate time evolution under H_heis3 Hamiltonian
        for _ in range(trotter_steps):
            qc.append(Trot_gate, q_regs)
    else: 
        qc.append(Trot_gate, q_regs)

    # Generate state tomography circuits to evaluate fidelity of simulation
    #st_qcs = state_tomography_circuits(qc, q_regs)
    st_qcs = qc
    
    # Display circuit for confirmation
    #display(st_qcs[-1].decompose().decompose().decompose().decompose().draw())  # view decomposition of trotter gates
    #display(st_qcs[-1].draw())  # only view trotter gates

    shots = 2 # 8192
    reps = 1

    meas_fitter = errorMitigation(qr, q_regs, backend, "complete")

    jobs = []
    for _ in range(reps):
        # Execute
        job = execute(st_qcs, backend, shots=shots)
        jobs.append(job)
    
    for job in jobs:
        job_monitor(job)
        try:
            if job.error_message() is not None:
                print(job.error_message())
        except:
            pass
        
    # Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
    def state_tomo(result, st_qcs, target_matrix):
        # The expected final state; necessary to determine state tomography fidelity
        target_state = target_matrix
        # Fit state tomography results
        tomo_fitter = StateTomographyFitter(result, st_qcs)
        rho_fit = tomo_fitter.fit(method='lstsq')
        # Compute fidelity
        fid = state_fidelity(rho_fit, target_state)
        return fid

    # Compute tomography fidelities for each repetition
    fids = []
    mit_fids = []
    for job in jobs:
        fids.append(state_tomo(job.result(), st_qcs, target_matrix))
        job = meas_fitter.filter.apply(job.result(), method='least_squares')
        mit_fids.append(state_tomo(job, st_qcs, target_matrix))
        
    properties = backend.properties()
    if properties == None:
        properties = 'NaN'
    else:
        properties = properties.backend_name
    print()
    print("Loaded: ", model_name)
    print('\nNo mitigation')
    print('F({}) = {:.3f} \u00B1 {:.3f}'.format(properties, np.mean(fids), np.std(fids)))
    print('\nComplete error mitigation')
    print('F({}) = {:.3f} \u00B1 {:.3f}\n'.format(properties, np.mean(mit_fids), np.std(mit_fids)))
        
    # Share tomography fidelity of discord to compete and collaborate with other students
    print('state tomography fidelity on ' + str(backend) + ' = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
    return qc
