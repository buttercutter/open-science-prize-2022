## Import functions that the stretched RZX will need
from qiskit.transpiler import PassManager
from qiskit import transpile

from qiskit.transpiler.passes.calibration.base_builder import CalibrationBuilder

from abc import abstractmethod
from typing import List, Union
import warnings

import math
import numpy as np

from qiskit.providers.backend import Backend
from qiskit.providers.backend import BackendV1
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    Play,
    Delay,
    ShiftPhase,
    Schedule,
    ScheduleBlock,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
)
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap, CalibrationPublisher
from qiskit.pulse.instructions.instruction import Instruction as PulseInst
from qiskit.transpiler.basepasses import TransformationPass

## Define a custom transpiler pass that modifies RZXCalibrationBuilderNoEcho to include pulse stretching for ZNE
class Stretched_RZXCalibrationBuilder(CalibrationBuilder):
    """"
    Modified version of RZXCalibrationBuilder to allow for stretched gates
    """

    def __init__(
        self,
        backend: Union[Backend, BackendV1] = None,
        instruction_schedule_map: InstructionScheduleMap = None,
        qubit_channel_mapping: List[List[str]] = None,
    ):
        """
        Initializes a RZXGate calibration builder.

        Args:
            backend: DEPRECATED a backend object to build the calibrations for.
                Use of this argument is deprecated in favor of directly
                specifying ``instruction_schedule_map`` and
                ``qubit_channel_map``.
            instruction_schedule_map: The :obj:`InstructionScheduleMap` object representing the
                default pulse calibrations for the target backend
            qubit_channel_mapping: The list mapping qubit indices to the list of
                channel names that apply on that qubit.

        Raises:
            QiskitError: if open pulse is not supported by the backend.
        """
        super().__init__()
        if backend is not None:
            warnings.warn(
                "Passing a backend object directly to this pass (either as the first positional "
                "argument or as the named 'backend' kwarg is deprecated and will no long be "
                "supported in a future release. Instead use the instruction_schedule_map and "
                "qubit_channel_mapping kwargs.",
                DeprecationWarning,
                stacklevel=2,
            )

            if not backend.configuration().open_pulse:
                raise QiskitError(
                    "Calibrations can only be added to Pulse-enabled backends, "
                    "but {} is not enabled with Pulse.".format(backend.name())
                )
            self._inst_map = backend.defaults().instruction_schedule_map
            self._channel_map = backend.configuration().qubit_channel_mapping

        else:
            if instruction_schedule_map is None or qubit_channel_mapping is None:
                raise QiskitError("Calibrations can only be added to Pulse-enabled backends")

            self._inst_map = instruction_schedule_map
            self._channel_map = qubit_channel_mapping

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, RZXGate)


    @staticmethod
    def rescale_cr_inst(instruction: Play, theta: float, stretch_factor: float, sample_mult: int = 16) -> Play:
        """
        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.
            stretch_factor: the factor to stretch the pulse by (while preserving area)

        Returns:
            qiskit.pulse.Play: The play instruction with the stretched compressed
                GaussianSquare pulse.

        Raises:
            QiskitError: if the pulses are not GaussianSquare.
        """

        pulse_ = instruction.pulse
        if isinstance(pulse_, GaussianSquare):
            amp = pulse_.amp
            width = pulse_.width
            sigma = pulse_.sigma
            n_sigmas = (pulse_.duration - width) / sigma

            # The error function is used because the Gaussian may have chopped tails.
            gaussian_area = abs(amp) * sigma * np.sqrt(2 * np.pi) * math.erf(n_sigmas)
            area = gaussian_area + abs(amp) * width

            target_area = abs(theta) / (np.pi / 2.0) * area
            sign = theta / abs(theta)

            if target_area > gaussian_area:
                width = (target_area - gaussian_area) / abs(amp) # stretches width
                duration = math.ceil((width + n_sigmas * sigma) / sample_mult ) * sample_mult # stretches duration
                
                # set new parameters with stretch factor 
                new_width = stretch_factor * width
                new_sigma = stretch_factor * sigma
                new_duration = math.ceil((new_width + n_sigmas * new_sigma) / sample_mult ) * sample_mult
                
                # return GaussianSquare with stretch factor
                return Play(
                    GaussianSquare(amp=sign * amp / stretch_factor, width=new_width, sigma=new_sigma, duration=new_duration),
                    channel=instruction.channel,
                )
            else:
                # this branch is irrelevant for our RZX stretching (relevant only for very small theta)
                amp_scale = sign * target_area / gaussian_area
                duration = math.ceil(n_sigmas * sigma / sample_mult) * sample_mult
                return Play(
                    GaussianSquare(amp=amp * amp_scale, width=0, sigma=sigma, duration=duration),
                    channel=instruction.channel,
                )
        else:
            raise QiskitError("RZXCalibrationBuilder only stretches/compresses GaussianSquare.")


        def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
            """Builds the calibration schedule for the RZXGate(theta) with echos.

            Args:
                node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
                qubits: List of qubits for which to get the schedules. The first qubit is
                    the control and the second is the target.

            Returns:
                schedule: The calibration schedule for the RZXGate(theta).

            Raises:
                QiskitError: if the control and target qubits cannot be identified or the backend
                    does not support cx between the qubits.
            """

            theta = node_op.params[0]
            q1, q2 = qubits[0], qubits[1]

            if not self._inst_map.has("cx", qubits):
                raise QiskitError(
                    "This transpilation pass requires the backend to support cx "
                    "between qubits %i and %i." % (q1, q2)
                )

            cx_sched = self._inst_map.get("cx", qubits=(q1, q2))
            rzx_theta = Schedule(name="rzx(%.3f)" % theta)
            rzx_theta.metadata["publisher"] = CalibrationPublisher.QISKIT

            if theta == 0.0:
                return rzx_theta

            crs, comp_tones = [], []
            control, target = None, None

            for time, inst in cx_sched.instructions:

                # Identify the CR pulses.
                if isinstance(inst, Play) and not isinstance(inst, ShiftPhase):
                    if isinstance(inst.channel, ControlChannel):
                        crs.append((time, inst))

                # Identify the compensation tones.
                if isinstance(inst.channel, DriveChannel) and not isinstance(inst, ShiftPhase):
                    if isinstance(inst.pulse, GaussianSquare):
                        comp_tones.append((time, inst))
                        target = inst.channel.index
                        control = q1 if target == q2 else q2

            if control is None:
                raise QiskitError("Control qubit is None.")
            if target is None:
                raise QiskitError("Target qubit is None.")

            echo_x = self._inst_map.get("x", qubits=control)

            # Build the schedule

            # Stretch/compress the CR gates and compensation tones
            cr1 = self.rescale_cr_inst(crs[0][1], theta, stretch_factor) # with area preserving stretch_factor
            cr2 = self.rescale_cr_inst(crs[1][1], theta, stretch_factor) # with area preserving stretch_factor

            if len(comp_tones) == 0:
                comp1, comp2 = None, None
            elif len(comp_tones) == 2:
                comp1 = self.rescale_cr_inst(comp_tones[0][1], theta, stretch_factor)# with area preserving stretch_factor
                comp2 = self.rescale_cr_inst(comp_tones[1][1], theta, stretch_factor)# with area preserving stretch_factor
            else:
                raise QiskitError(
                    "CX must have either 0 or 2 rotary tones between qubits %i and %i "
                    "but %i were found." % (control, target, len(comp_tones))
                )

            # Build the schedule for the RZXGate
            rzx_theta = rzx_theta.insert(0, cr1)

            if comp1 is not None:
                rzx_theta = rzx_theta.insert(0, comp1)

            rzx_theta = rzx_theta.insert(comp1.duration, echo_x)
            time = comp1.duration + echo_x.duration
            rzx_theta = rzx_theta.insert(time, cr2)

            if comp2 is not None:
                rzx_theta = rzx_theta.insert(time, comp2)

            time = 2 * comp1.duration + echo_x.duration
            rzx_theta = rzx_theta.insert(time, echo_x)

            # Reverse direction of the ZX with Hadamard gates
            if control == qubits[0]:
                return rzx_theta
            else:
                rzc = self._inst_map.get("rz", [control], np.pi / 2)
                sxc = self._inst_map.get("sx", [control])
                rzt = self._inst_map.get("rz", [target], np.pi / 2)
                sxt = self._inst_map.get("sx", [target])
                h_sched = Schedule(name="hadamards")
                h_sched = h_sched.insert(0, rzc)
                h_sched = h_sched.insert(0, sxc)
                h_sched = h_sched.insert(sxc.duration, rzc)
                h_sched = h_sched.insert(0, rzt)
                h_sched = h_sched.insert(0, sxt)
                h_sched = h_sched.insert(sxc.duration, rzt)
                rzx_theta = h_sched.append(rzx_theta)
                return rzx_theta.append(h_sched)



class Stretched_RZXCalibrationBuilderNoEcho(Stretched_RZXCalibrationBuilder):
    """
    Modified version of RZXCalibrationBuilderNoEcho to allow for stretched gates
    """

    @staticmethod
    def _filter_control(inst: (int, Union["Schedule", PulseInst])) -> bool:
        """
        Looks for Gaussian square pulses applied to control channels.

        Args:
            inst: Instructions to be filtered.

        Returns:
            match: True if the instruction is a Play instruction with
                a Gaussian square pulse on the ControlChannel.
        """
        if isinstance(inst[1], Play):
            if isinstance(inst[1].pulse, GaussianSquare) and isinstance(
                inst[1].channel, ControlChannel
            ):
                return True

        return False

    @staticmethod
    def _filter_drive(inst: (int, Union["Schedule", PulseInst])) -> bool:
        """
        Looks for Gaussian square pulses applied to drive channels.

        Args:
            inst: Instructions to be filtered.

        Returns:
            match: True if the instruction is a Play instruction with
                a Gaussian square pulse on the DriveChannel.
        """
        if isinstance(inst[1], Play):
            if isinstance(inst[1].pulse, GaussianSquare) and isinstance(
                inst[1].channel, DriveChannel
            ):
                return True

        return False

    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Builds the calibration schedule for the RZXGate(theta) without echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: If the control and target qubits cannot be identified, or the backend
                does not support a cx gate between the qubits, or the backend does not natively
                support the specified direction of the cx.
        """
        theta = node_op.params[0]
        q1, q2 = qubits[0], qubits[1]

        if not self._inst_map.has("cx", qubits):
            raise QiskitError(
                "This transpilation pass requires the backend to support cx "
                "between qubits %i and %i." % (q1, q2)
            )

        cx_sched = self._inst_map.get("cx", qubits=(q1, q2))
        rzx_theta = Schedule(name="rzx(%.3f)" % theta)
        rzx_theta.metadata["publisher"] = CalibrationPublisher.QISKIT

        if theta == 0.0:
            return rzx_theta

        control, target = None, None

        for _, inst in cx_sched.instructions:
            # Identify the compensation tones.
            if isinstance(inst.channel, DriveChannel) and isinstance(inst, Play):
                if isinstance(inst.pulse, GaussianSquare):
                    target = inst.channel.index
                    control = q1 if target == q2 else q2

        if control is None:
            raise QiskitError("Control qubit is None.")
        if target is None:
            raise QiskitError("Target qubit is None.")

        if control != qubits[0]:
            raise QiskitError(
                "RZXCalibrationBuilderNoEcho only supports hardware-native RZX gates."
            )

        # Get the filtered Schedule instructions for the CR gates and compensation tones.
        crs = cx_sched.filter(*[self._filter_control]).instructions
        rotaries = cx_sched.filter(*[self._filter_drive]).instructions

        # Stretch/compress the CR gates and compensation tones.
        cr = self.rescale_cr_inst(crs[0][1], 2 * theta, stretch_factor)
        rot = self.rescale_cr_inst(rotaries[0][1], 2 * theta, stretch_factor)

        # Build the schedule for the RZXGate without the echos.
        rzx_theta = rzx_theta.insert(0, cr)
        rzx_theta = rzx_theta.insert(0, rot)
        rzx_theta = rzx_theta.insert(0, Delay(cr.duration, DriveChannel(control)))

        return rzx_theta
    
def get_stretched_pulses(st_qcs, backend, stretch_factor_given):
    '''
    Args:
        st_qc: Quantum Circuits objects to transpile
        backend: the backend for this pulse generation
        stretch_factor_given: the factor by which the RZX pulses will be streched by (area preserved)
    
    Returns:
        qc_pulse_efficient: the transpiled pulses with RZX pulses stretched by stretch_factor (area preserved)
    '''
    
    global stretch_factor # somehow necessary to set to global to avoid some bugs
    stretch_factor = stretch_factor_given

    pm = PassManager([Stretched_RZXCalibrationBuilderNoEcho(backend)])
    qc_pulse_efficient = pm.run(st_qcs)
    qc_pulse_efficient = transpile(qc_pulse_efficient, backend)

    return qc_pulse_efficient