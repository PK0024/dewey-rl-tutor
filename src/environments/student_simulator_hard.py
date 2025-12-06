"""Hard Student Simulator (for transfer learning)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from environments.student_simulator import StudentSimulator

class HardStudentSimulator(StudentSimulator):
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed, options)
        self.true_skill = 0.6 + 0.2 * (self.true_skill - 0.25) / 0.5
        return self._get_observation(), info
