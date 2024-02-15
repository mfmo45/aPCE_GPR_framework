"""

"""
import numpy as np
import math


class GRID:
    """
    Class saves the grid properties for a model setup:

    Parameters:
        grid_size: list[2,1], with number of cells in the [x, y] directions
        grid_distance: list[2, 1], with length of each cell in the [x, y] direction
        covariance_data : dict
            {name : string,  options: 'exponential'
            mean : float,    RF value mean
            variance : float,   RF variance
            correlation_length}   list/array [x, y]

    Attributes:
        self.grid_size = list[2,1],
            with number of cells in the [x, y] directions
        self.grid_distance = list[2, 1],
            with length of each cell in the [x, y] direction
        self.covariance_data : dict

        self.n_cells = int,
            number of cells in the grid
        self.domain_size = list[2, 1],
            total length of domain in the [x, y] distance
        self.x_grid = np.array [n_x, n_y]
            with x coordinates of each cell, where n_x and n_y are the number of cells in
         each direction
        self.y_grid = np.array[n_x, n_y]
            with y coordinates of each cell, where n_x and n_y are the number of cells in each direction
        self.x_vec = np.array[n_cells, 1],
            with x coordinates of each cell, arranged column-wise (columns from self.x_grid were stacked on top of
            each other, in order)
        self.y_vec = np.array[n_cells, 1],
        with y coordinates of each cell, arranged column-wise (columns from self.x_grid were stacked on top of each other, in order)

        self.cov_matrix : array [n_cells, n_cells]
            with the covariance matrix, based on the full grid geometry.

    ToDO: The exponential covariance function is used here. More can be added, along with additional functions and
    input parameters.

    """
    def __init__(self, grid_size, grid_distance, covariance_data):
        self.grid_size = grid_size
        self.grid_distance = grid_distance
        self.covariance_data = covariance_data

        self.grid_points = None
        self.n_cells = None
        self.n_pts = None
        self.domain_size = None
        self.x_grid = None
        self.y_grid = None
        self.x_vec = None
        self.y_vec = None

        self.cov_matrix = None

        self.calculate_attributes()
        self.get_grid()

    def calculate_attributes(self):
        """ Function calculates basic attributes"""
        self.n_cells = int(self.grid_size[0] * self.grid_size[1])
        self.domain_size = np.array(self.grid_size) * np.array(self.grid_distance)

        self.grid_points = np.array([[int(self.grid_size[0]+1), int(self.grid_size[0]+1)]])
        self.n_pts = int((self.grid_size[0]+1) * (self.grid_size[1]+1))

    def get_grid(self):
        """Function generates 2 2D grids with the x and y coordinates of each cell in the grid"""
        # Create grids with x and y locations
        x_loc = np.arange(0, self.domain_size[0], self.grid_distance[0])
        y_loc = np.arange(0, self.domain_size[1], self.grid_distance[1])
        self.x_grid, self.y_grid = np.meshgrid(x_loc, y_loc)

        # Create vector with grid locations (stacks elements column-wise):
        self.x_vec = np.reshape(self.x_grid, (self.n_cells, 1), order="F")
        self.y_vec = np.reshape(self.y_grid, (self.n_cells, 1), order="F")

    def get_covariance(self):
        """
        Function generates covariance function based on Grid data.
        ToDO: add additional covariance functions
        """
        if self.covariance_data['log_rf']:
            self.covariance_data['mean'] = math.log(self.covariance_data['mean'])

        if not isinstance(self.covariance_data['corr_length'], np.ndarray):
            self.covariance_data['corr_length'] = self.covariance_data['corr_length']

        if self.covariance_data['name'] == 'exponential':
            self.cov_matrix = self._exponential_covariance()

        elif self.covariance_data['name'] == 'exponential_type2':
            self.cov_matrix = self._exponential_covariance_type2()

    def _exponential_covariance(self):
        """
        Function calculates the covariance matrix using an exponential kernel. Function needs the grid coordinates (in
        vector form), and the covariance correlation length. It uses a variance of 1 for this calculation. The variance
        must then be multiplied by the covariance matrix outside the function.

        Returns: np.array[grid] with exponential covariance matrix with variance of 1.
        """
        # For exponential covariance:
        cov_matrix = np.zeros((self.n_cells, self.n_cells))
        for i in range(0, self.n_cells):  # each row, each point i
            x_i = np.power(np.divide(np.subtract(self.x_vec[i], self.x_vec), self.covariance_data['corr_length'][0]),
                           2)
            y_i = np.power(np.divide(np.subtract(self.y_vec[i], self.y_vec), self.covariance_data['corr_length'][1]),
                           2)
            cov_matrix[i, :] = (np.sqrt(np.add(x_i, y_i)))[:, 0]

        cov_matrix = self.covariance_data['var'] * np.exp(-1 * cov_matrix)

        return cov_matrix

    def _exponential_covariance_type2(self):
        """
        Function calculates the covariance matrix using an exponential kernel, considering the absolute value of 'h'.
         Function needs the grid coordinates (in vector form), and the covariance correlation length.
         It uses a variance of 1 for this calculation.

        Returns: np.array[grid]
            with covariance matrix.
        """
        # For exponential covariance:
        cov_matrix = np.zeros((self.n_cells, self.n_cells))
        for i in range(0, self.n_cells):  # each row, each point i
            x_i = np.divide(np.abs(np.subtract(self.x_vec[i], self.x_vec)), self.covariance_data['corr_length'][0])
            y_i = np.divide(np.abs(np.subtract(self.y_vec[i], self.y_vec)), self.covariance_data['corr_length'][1])
            cov_matrix[i, :] = np.add(x_i, y_i)[:, 0]

        cov_matrix = self.covariance_data['var'] * np.exp(-1 * cov_matrix)

        return cov_matrix

    def update(self, new_grid_size=None, new_grid_distance=None):
        """
        Function allows to change the grid_size and/or grid_distance to generate a new grid, so it recalculates all
        attributes.
        """
        if new_grid_size is not None:
            self.grid_size = new_grid_size
        if new_grid_distance is not None:
            self.grid_distance = new_grid_distance
        self.calculate_attributes()
        self.get_grid()


class KLD(GRID):
    """
    Class generates Random Fields using the Karhunen-Loève Decomposition.
    As default, the random field is generated with the full grid, meaning that all eigenvalues are used to generate the
    random field. The user can, however, set a truncation value to generate it with a smaller number of eigenvalues.

    The eigen-problem is solved using the numpy library, specifically numpy.linalg.eigh, which is for positive,
    semi-definite covariance, as is the case here.

    Parameters:
        n_truncation: float, with number of eigenvalues to consider - by default it is equal to None, and thus a value
        equivalent to the number of cells will be assigned to it.

    Attributes:
        self.truncation_n = truncation_n

        self.eigen_values = np.array[truncation_n, 1] with eigenvalues corresponding to self.cov_matrix
        self.eigen_vectors = np.array[n_cells, truncation_n] with eigenvectors of eigen_values

    Notes:
        Currently: only for constant covariance hyper-parameters, and constant mean.
    ToDo: add variations in hyper-parameters:
    ToDO: When estimating eigen, only save for the n_truncation value
    """
    def __init__(self, grid_size, grid_distance, covariance_data, generate_truncated_rf=False,
                 n_truncation=None):

        super(KLD, self).__init__(grid_size=grid_size, grid_distance=grid_distance, covariance_data=covariance_data)

        # for KLD:
        self.truncation_n = n_truncation
        self.get_truncated_rf = generate_truncated_rf

        self.eigen_values = None
        self.eigen_vectors = None

    def get_eigen_data(self):
        """
        Function returns the eigenvalues and eigenvector of the covariance matrix, ordered according to descending
        eigenvalues.
        """
        # Get truncation N:
        if self.truncation_n is None:
            self.truncation_n = self.n_cells
        # Calculate eigenvalues and eigen vectors --------------------------
        eig_values, eig_vectors = np.linalg.eigh(self.cov_matrix)

        # get index in descending order
        idx = np.argsort(eig_values)[::-1]

        # order eigenvalues and eigenvectors in descending order
        self.eigen_values = eig_values[idx].reshape((eig_values.shape[0], 1))
        self.eigen_vectors = eig_vectors[:, idx]

    def rf_with_kld(self, eigen_coeff):
        """
        Function generates the ln(K) field using KL Decomposition. It first truncates the eigenvalues, and then uses
        the RV to generate the RF. It calculates the RF assuming a mean of zero. The mean must be added outside of the
        function.
        Returns: np.array[n_cells, 1], ln(K) values for each grid, in vector form
        """
        # truncate:
        if eigen_coeff.shape[0] != self.truncation_n:
            trunc = eigen_coeff.shape[0]
        else:
            trunc = self.truncation_n

        e_values = self.eigen_values[0:trunc]
        e_vectors = self.eigen_vectors[:, 0:trunc]
        r_v = eigen_coeff[0:trunc, :]

        # # Option 2:
        # e_values_2 = np.diag(e_values[:, 0])
        # k_vector_2 = np.dot(np.dot(e_vectors, np.sqrt(e_values_2)), r_v)

        step_1 = np.multiply(np.sqrt(e_values), r_v)
        k_vector = np.dot(e_vectors, step_1)
        return k_vector

    def generate_rf(self, n_realizations=1, eigen_coeff=None):
        """
        Function generates a random field using the KL decomposition method. If no random variables are given,
        it first samples 'n_realizations' N(0, 1).

        Args:
            n_realizations: int
                Number of RF to generate. Default is 1 (will be made to match the size of the random variables)
            eigen_coeff: array [n_truncation, n_realizations]
                N(0,1) random variables, to be used as eigenvalue coefficients.

        Returns: array [n_cells, n_realization]
            'n_realization' RF, generated using the KLD method.

        """
        # If no random variables are given, estimate 'n_realizations' random samples for eigenvalue coefficients
        if eigen_coeff is None:
            eigen_coeff = np.random.normal(loc=0, scale=1, size=(self.truncation_n, n_realizations))
        else:
            if n_realizations != eigen_coeff.shape[1]:
                n_realizations = eigen_coeff.shape[1]

        # Generate random field:
        rf = self.covariance_data['mean'] + self.rf_with_kld(eigen_coeff=eigen_coeff)

        return rf