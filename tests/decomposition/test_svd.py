from vcglearn import SVD
import pytest
from tests.core import BaseTest

@pytest.fixture
def svd():
	return SVD()

class TestSVD(BaseTest):
	test_name = "_svd"
	test_area = "decomposition"

	def test_compute_ata(self, svd):
		A = np.array([
			[1, 2],
			[3, 4]
		], dtype=float)

		A_result = np.array([
			[10, 14],
			[14, 20]
		])

		ata_A = svd._compute_ata(A)

		assert ata_A == A_result

		rank_def_A = np.array([
			[1, 2, 3 ],
			[2, 4, 6 ],
			[3, 6, 9 ],
			[4, 8, 12]
		], dtype=float)

		rank_def_A_result = np.array([
			[30, 60, 90  ],
			[60, 120, 180],
			[90, 180, 270]
		], dtype=float)

		ata_rank_def_A = svd._compute_ata(rank_def_A)

		assert ata_rank_def_A == rank_def_A_result

	def test_flip_eigval(self, svd):
		pass

	def test_flip_eigvec(self, svd):
		pass

	def test_compute_eigv(self, svd):
		pass

	def test_compute_singv(self, svd):
		pass

	def test_compute_sigma(self, svd):
		pass

	def test_fit(self, svd):
		pass

	def test_predict(self, svd):
		pass

	def test_fit_predict(self, svd):
		pass