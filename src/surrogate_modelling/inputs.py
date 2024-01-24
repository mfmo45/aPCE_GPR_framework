#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Input:
    """
    A class to define the uncertain input parameters.

    Attributes
    ----------
    Marginals : obj
        Marginal objects. See `inputs.Marginal`.
    Rosenblatt : bool
        If Rossenblatt transformation is required for the dependent input
        parameters.

    Examples
    -------
    Marginals can be defined as following:

    >>> Inputs.add_marginals()
    >>> Inputs.Marginals[0].name = 'X_1'
    >>> Inputs.Marginals[0].dist_type = 'uniform'
    >>> Inputs.Marginals[0].parameters = [-5, 5]

    If there is no common data is avaliable, the input data can be given
    as following:

    >>> Inputs.add_marginals()
    >>> Inputs.Marginals[0].name = 'X_1'
    >>> Inputs.Marginals[0].input_data = input_data
    """
    poly_coeffs_flag = True

    def __init__(self):
        self.Marginals = []
        self.Rosenblatt = False

    def add_marginals(self):
        """
        Adds a new Marginal object to the input object.

        Returns
        -------
        None.

        """
        self.Marginals.append(Marginal())


# Nested class
class Marginal:
    """
    An object containing the specifications of the marginals for each uncertain
    parameter.

    Attributes
    ----------
    name : string
        Name of the parameter. The default is `'$x_1$'`.
    dist_type : string
        Name of the distribution. The default is `None`.
    parameters : list
        List of the parameters corresponding to the distribution type. The
        default is `None`.
    input_data : array
        Available input data. The default is `[]`.
    moments : list
        List of the moments.
    """

    def __init__(self):
        self.name = '$x_1$'
        self.dist_type = None
        self.parameters = None
        self.input_data = []
        self.moments = None
