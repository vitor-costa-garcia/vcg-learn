import numpy as np
import pytest

from vcglearn import SVD
from tests.core import BaseTest

@pytest.fixture
def svd():
	return SVD()

class TestSVD(BaseTest):
	test_name = "_svd"
	test_area = "decomposition"

	def test_compute_ata(self, svd):
		# Full-rank matrix
		A = np.array([
			[1, 2],
			[3, 4]
		], dtype=float)

		A_result = np.array([
			[10, 14],
			[14, 20]
		])

		svd_ata_A = svd._compute_ata(A)

		assert np.array_equal(svd_ata_A, A_result)

		# Rank-deficient matrix
		rank_def_A = np.array([
			[1, 2, 3 ],
			[2, 4, 6 ],
			[3, 6, 9 ],
			[4, 8, 12]
		], dtype=float)

		rank_def_A_result = np.array([
			[30, 60,  90 ],
			[60, 120, 180],
			[90, 180, 270]
		], dtype=float)

		svd_ata_rank_def_A = svd._compute_ata(rank_def_A)

		assert np.array_equal(svd_ata_rank_def_A, rank_def_A_result)

		#Zero matrix
		zeros_A = np.zeros((2, 3))
		zeros_A_result = np.zeros((3, 3))

		svd_ata_zeros_A = svd._compute_ata(zeros_A)

		assert np.array_equal(svd_ata_zeros_A, zeros_A_result)

		#Empty matrix
		empty_A = np.array([])
		empty_A_result = 0.0 #Sum of 0 elements is 0

		svd_ata_empty_A = svd._compute_ata(empty_A)
		print(svd_ata_empty_A)

		assert np.array_equal(svd_ata_empty_A, empty_A_result)


	def test_flip_eigval(self, svd):
		# Size N array
		A_1 = np.array([1,2,3,4,5,6])
		A_result_1 = np.array([6,5,4,3,2,1])

		svd_flip_eigval_1 = svd._flip_eigval(A_1)

		assert np.array_equal(svd_flip_eigval_1, A_result_1)

		# Size 1 array
		A_2 = np.array([1])
		A_result_2 = np.array([1])

		svd_flip_eigval_2 = svd._flip_eigval(A_2)

		assert np.array_equal(svd_flip_eigval_2, A_result_2)

		# Empty array
		A_3 = np.array([])
		A_result_3 = np.array([])

		svd_flip_eigval_3 = svd._flip_eigval(A_3)

		assert np.array_equal(svd_flip_eigval_3, A_result_3)

	def test_flip_eigvec(self, svd):
		pass

	def test_clip_eigval(self, svd):
		pass

	def test_compute_eigv(self, svd):
		pass

	def test_compute_singv(self, svd):
		pass

	def test_tol_singv(self, svd):
		pass

	def test_clip_v(self, svd):
		pass

	def test_compute_u(self, svd):
		pass

	def test_compute_sigma(self, svd):
		pass

	def test_fit(self, svd):
		pass

	def test_predict(self, svd):
		pass

	def test_fit_predict(self, svd):
		pass