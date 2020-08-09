"""
VASP calculation.

-----------------
The calculation class that prepares a specific VASP calculation.
"""
#encoding: utf-8
# pylint: disable=abstract-method
# explanation: pylint wrongly complains about (aiida) Node not implementing query
from aiida.plugins import DataFactory

from aiida_vasp.parsers.file_parsers.incar import IncarParser
from aiida_vasp.parsers.file_parsers.potcar import MultiPotcarIo
from aiida_vasp.parsers.file_parsers.poscar import PoscarParser
from aiida_vasp.parsers.file_parsers.kpoints import KpointsParser
from aiida_vasp.utils.aiida_utils import get_data_node, get_data_class
from aiida_vasp.calcs.base import VaspCalcBase
from aiida_vasp.utils.inheritance import update_docstring

PARAMETER_CLS = DataFactory('dict')
SINGLEFILE_CLS = DataFactory('singlefile')

_IMMIGRANT_EXTRA_KWARGS = """
vasp.vasp specific kwargs:

:param use_chgcar: bool, if True, read the CHGCAR file (has to exist) and convert it to an input node.
:param use_wavecar: bool, if True, read the WAVECAR file (has to exist) and convert it to an input node.
"""


@update_docstring('immigrant', _IMMIGRANT_EXTRA_KWARGS, append=True)
class VaspCalculation(VaspCalcBase):
    """
    General-purpose VASP calculation.

    ---------------------------------
    By default retrieves only the 'OUTCAR', 'vasprun.xml', 'EIGENVAL', 'DOSCAR'
    and Wannier90 input / output files. These files are deleted after parsing.
    Additional retrieve files can be specified via the
    ``settings['ADDITIONAL_RETRIEVE_TEMPORARY_LIST']`` input. In addition, if you want to keep
    any files after parsing, put them in ``settings['ADDITIONAL_RETRIEVE_LIST']`` which is empty
    by default.

    Floating point precision for writing POSCAR files can be adjusted using
    ``settings['poscar_precision']``, default: 10

    The following assumes you are familiar with the AiiDA data structures and
    how to set up and run an AiiDA calculation in general.

    Example usage::

        from aiida.orm import CalculationFactory, DataFactory
        from aiida.work import submit

        proc = CalculationFactory('vasp.vasp').process()
        inputs = proc.get_inputs_template()
        inputs.parameter = <Dict with INCAR params>
        inputs.structure = <StructureData>
        inputs.kpoints = <KpointsData>
        inputs.settings = <Dict with parser settings etc.>
        inputs.potential = DataFactory('vasp.potcar').get_potcars_from_structure(structure, ...)
        inputs.code = <Code representing vasp on your cluster>

        submit(proc, **inputs)

    Which is very similar to the workchain example.

    """

    _ALWAYS_RETRIEVE_LIST = ['CONTCAR', 'OUTCAR', 'vasprun.xml', 'EIGENVAL', 'DOSCAR', 'wannier90*']
    _query_type_string = 'vasp.vasp'
    _plugin_type_string = 'vasp.vasp'

    @classmethod
    def define(cls, spec):
        super(VaspCalculation, cls).define(spec)
        # Define the inputs.
        # options is passed automatically.
        spec.input('parameters', valid_type=get_data_class('dict'), help='The VASP input parameters (INCAR).')
        spec.input('structure', valid_type=(get_data_class('structure'), get_data_class('cif')), help='The input structure (POSCAR).')
        # Need namespace on this as it should also accept keys that are of `kind`. These are unknown
        # until execution.
        spec.input_namespace('potential', valid_type=get_data_class('vasp.potcar'), help='The potentials (POTCAR).', dynamic=True)
        spec.input('kpoints', valid_type=get_data_class('array.kpoints'), help='The kpoints to use (KPOINTS).')
        spec.input('charge_density', valid_type=get_data_class('vasp.chargedensity'), required=False, help='The charge density. (CHGCAR)')
        spec.input('wavefunctions',
                   valid_type=get_data_class('vasp.wavefun'),
                   required=False,
                   help='The wave function coefficients. (WAVECAR)')
        spec.input('settings', valid_type=get_data_class('dict'), required=False, help='Additional parameters not related to VASP itself.')
        spec.input('metadata.options.parser_name', default='vasp.vasp')

        # Define outputs.
        # remote_folder and retrieved are passed automatically
        spec.output('misc',
                    valid_type=get_data_class('dict'),
                    help='The output parameters containing smaller quantities that do not depend on system size.')
        spec.output('structure', valid_type=get_data_class('structure'), required=False, help='The output structure.')
        spec.output('kpoints', valid_type=get_data_class('array.kpoints'), required=False, help='The output k-points.')
        spec.output('trajectory', valid_type=get_data_class('array.trajectory'), required=False, help='The output trajectory data.')
        spec.output('chgcar', valid_type=get_data_class('vasp.chargedensity'), required=False, help='The output charge density.')
        spec.output('wavecar',
                    valid_type=get_data_class('vasp.wavefun'),
                    required=False,
                    help='The output file containing the plane wave coefficients.')
        spec.output('bands', valid_type=get_data_class('array.bands'), required=False, help='The output band structure.')
        spec.output('forces', valid_type=get_data_class('array'), required=False, help='The output forces.')
        spec.output('stress', valid_type=get_data_class('array'), required=False, help='The output stress.')
        spec.output('dos', valid_type=get_data_class('array'), required=False, help='The output dos.')
        spec.output('occupancies', valid_type=get_data_class('array'), required=False, help='The output band occupancies.')
        spec.output('energies', valid_type=get_data_class('array'), required=False, help='The output total energies.')
        spec.output('projectors', valid_type=get_data_class('array'), required=False, help='The output projectors of decomposition.')
        spec.output('dielectrics', valid_type=get_data_class('array'), required=False, help='The output dielectric functions.')
        spec.output('born_charges', valid_type=get_data_class('array'), required=False, help='The output Born effective charges.')
        spec.output('hessian', valid_type=get_data_class('array'), required=False, help='The output Hessian matrix.')
        spec.output('dynmat', valid_type=get_data_class('array'), required=False, help='The output dynamical matrix.')
        spec.output('site_magnetization', valid_type=get_data_class('dict'), required=False, help='The output of the site magnetization')
        spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        spec.exit_code(350, 'ERROR_NO_RETRIEVED_FOLDER', message='the retrieved folder data node could not be accessed.')
        spec.exit_code(351,
                       'ERROR_NO_RETRIEVED_TEMPORARY_FOLDER',
                       message='the retrieved_temporary folder data node could not be accessed.')
        spec.exit_code(352, 'ERROR_CRITICAL_MISSING_FILE', message='a file that is marked by the parser as critical is missing.')
        
        # These are erros that can be retrived from _schedular.stdout
        spec.exit_code(400, "ERROR_TETRAHEDRON_NKPT", message="Tetrahedron method fails for NKPT<4")
        spec.exit_code(401, "ERROR_TETRAHEDRON_KMESH", message="Fatal error detecting k-mesh")
        spec.exit_code(402, "ERROR_TETRAHEDRON_KPOINT_MATCH", message="Fatal error: unable to match k-point")
        spec.exit_code(403, "ERROR_TETRAHEDRON_TETIRR", message="Routine TETIRR needs special values")
        spec.exit_code(404, "ERROR_TETRAHEDRON_KPOINT_NUM", message="Tetrahedron method fails (number of k-points < 4)")
        spec.exit_code(405, "ERROR_TETRAHEDRON_DENTET", message="DENTET")
        
        spec.exit_code(405, "ERROR_INVERSE_ROTATION_MATRIX", message="inverse of rotation matrix was not found (increase SYMPREC)")
        spec.exit_code(406, "ERROR_BRMIX", message="BRMIX: very serious problems")
        spec.exit_code(407, "ERROR_SUBSPACEMATRIX", message="WARNING: Sub-Space-Matrix is not hermitian in DAV")
        spec.exit_code(408, "ERROR_TETIRR", message="Routine TETIRR needs special values")
        spec.exit_code(409, "ERROR_INCORRECT_SHIFT", message="Could not get correct shifts")
        spec.exit_code(410, "ERROR_REAL_OPTLAY", message="REAL_OPTLAY: internal error")
        spec.exit_code(411, "ERROR_REAL_OPTLAY", message="REAL_OPT: internal ERROR")
        spec.exit_code(412, "ERROR_RSPHER", message="ERROR RSPHER")
        
        spec.exit_code(414, "ERROR_TOO_FEW_BANDS", message="TOO FEW BANDS")
        spec.exit_code(415, "ERROR_TRIPLE_PRODUCT", message="ERROR: the triple product of the basis vectors")
        spec.exit_code(416, "ERROR_ROT_MATRIX", message="Found some non-integer element in rotation matrix")
        spec.exit_code(417, "ERROR_BRIONS", message="BRIONS problems: POTIM should be increased")
        spec.exit_code(418, "ERROR_PRICEL", message="internal error in subroutine PRICEL")
        spec.exit_code(419, "ERROR_ZPOTRF", message="LAPACK: Routine ZPOTRF failed")
        spec.exit_code(420, "ERROR_AMIN", message="One of the lattice vectors is very long (>50 A), but AMIN")
        spec.exit_code(421, "ERROR_ZBRENT", message="ZBRENT: fatal internal in")
        spec.exit_code(422, "ERROR_ZBRENT", message="ZBRENT: fatal error in bracketing")
        spec.exit_code(423, "ERROR_PSSYEVX", message="ERROR in subspace rotation PSSYEVX")
        spec.exit_code(424, "ERROR_EDDRMM", message="WARNING in EDDRMM: call to ZHEGV failed")
        spec.exit_code(425, "ERROR_EDDDAV", message="Error EDDDAV: Call to ZHEGV failed")
        spec.exit_code(426, "ERROR_GRAD_NOT_ORTH", message="EDWAV: internal error, the gradient is not orthogonal")
        spec.exit_code(427, "ERROR_NICHT_KONV", message="ERROR: SBESSELITER : nicht konvergent")
        spec.exit_code(428, "ERROR_ZHEEV", message="ERROR EDDIAG: Call to routine ZHEEV failed!")
        spec.exit_code(429, "ERROR_ELF_KPAR", message="ELF: KPAR>1 not implemented")
        spec.exit_code(430, "ERROR_ELF_NCL", message="WARNING: ELF not implemented for non collinear case")
        spec.exit_code(431, "ERROR_RHOSYG", message="RHOSYG internal error")
        spec.exit_code(432, "ERROR_POSMAP", message="POSMAP internal error: symmetry equivalent atom not found")
        spec.exit_code(433, "ERROR_POINT_GROUP", message="Error: point group operation missing")
        spec.exit_code(434, "ERROR_ALIASING", message="WARNING: small aliasing (wrap around) errors must be expected")
        spec.exit_code(435, "ERROR_ALIASING_INCAR", message="Your FFT grids (NGX,NGY,NGZ) are not sufficient for an accurate")
        spec.exit_code(436, "ERROR_KPOINT_SYMM", message="Reciprocal lattice and k-lattice belong to different class of" " lattices.")

        # These are errors that can be retrieved from _schedulat.stderr --> Start from 450
        spec.exit_code(450, "ERROR_LRF_COMM", message="LRF_COMMUTATOR internal error")
        spec.exit_code(451, "ERROR_KPOINTS_TRANS", message="internal error in GENERATE_KPOINTS_TRANS: ")
        spec.exit_code(452, "ERROR_KPOINTS_TRANS", message="number of G-vector changed in star")
        spec.exit_code(453, "ERROR_OUT_OF_MEMORY", message="Allocation would exceed memory limit")

        # These are erros that can be spotted from vasprun.xml --> start from 460
        spec.exit_code(460, "ERROR_COVERGENCE_NOT_REACHED", message="Calculation is not converged")
        spec.exit_code(461, "ERROR_ELECTRONIC_CONVERGENCE_NOT_REACHED", message="Calculation is not converged electronically")
        spec.exit_code(462, "ERROR_IONIC_CONVERGENCE_NOT_REACHED", message="Calculation is not converged in ionic steps")

        spec.exit_code(1001, 'ERROR_PARSING_FILE_FAILED', message='parsing a file has failed.')
        spec.exit_code(1002, 'ERROR_NOT_ABLE_TO_PARSE_QUANTITY', message='the parser is not able to parse the requested quantity')

    def prepare_for_submission(self, tempfolder):
        """Add EIGENVAL, DOSCAR, and all files starting with wannier90 to the list of files to be retrieved."""
        calcinfo = super(VaspCalculation, self).prepare_for_submission(tempfolder)
        # Still need the exceptions in case settings is not defined on inputs
        # Check if we want to store all always retrieve files
        try:
            store = self.inputs.settings.get_attribute('ALWAYS_STORE', default=True)
        except AttributeError:
            store = True
        try:
            additional_retrieve_list = self.inputs.settings.get_attribute('ADDITIONAL_RETRIEVE_LIST', default=[])
        except AttributeError:
            additional_retrieve_list = []
        try:
            additional_retrieve_temp_list = self.inputs.settings.get_attribute('ADDITIONAL_RETRIEVE_TEMPORARY_LIST', \
                                                                               default=[])  # pylint: disable=invalid-name
        except AttributeError:
            additional_retrieve_temp_list = []
        if store:
            calcinfo.retrieve_list = list(set(self._ALWAYS_RETRIEVE_LIST + additional_retrieve_list))
            calcinfo.retrieve_temporary_list = additional_retrieve_temp_list  # pylint: disable=invalid-name
        else:
            calcinfo.retrieve_temporary_list = list(set(self._ALWAYS_RETRIEVE_LIST + additional_retrieve_temp_list))  # pylint: disable=invalid-name
            calcinfo.retrieve_list = additional_retrieve_list
        try:
            provenance_exclude_list = self.inputs.settings.get_attribute('PROVENANCE_EXCLUDE_LIST', default=[])
        except AttributeError:
            provenance_exclude_list = []
        # Always include POTCAR in the exclude list (not added to the repository)
        calcinfo.provenance_exclude_list = list(set(provenance_exclude_list + ['POTCAR']))

        return calcinfo

    def verify_inputs(self):
        super(VaspCalculation, self).verify_inputs()
        if not hasattr(self, 'elements'):
            self._prestore()

    def _prestore(self):
        """Set attributes prior to storing."""
        super(VaspCalculation, self)._prestore()
        setattr(self, 'elements', ordered_unique_list(self.inputs.structure.get_ase().get_chemical_symbols()))

    @property
    def _parameters(self):
        """Make sure all parameters are lowercase."""
        all_parameters = self.inputs.parameters.get_dict()
        try:
            return {k.lower(): v for k, v in all_parameters.items()}
        except KeyError:
            return {}

    def _need_kp(self):
        """
        Return wether an input kpoints node is needed or not.

        :return output:
            True if input kpoints node is needed
            (py:method::VaspCalculation.use_kpoints),
            False otherwise
        needs 'parameters' input to be set
        (py:method::VaspCalculation.use_parameters)
        """
        return not bool('kspacing' in self._parameters or 'kgamma' in self._parameters)

    def _need_chgcar(self):
        """
        Test wether an charge_densities input is needed or not.

        :return output:
            True if a chgcar file must be used
            (py:method::NscfCalculation.use_charge_densities),
            False otherwise
        needs 'parameters' input to be set
        (py:method::NscfCalculation.use_parameters)
        """
        ichrg_d = 0 if self._need_wavecar() else 2
        icharg = self._parameters.get('icharg', ichrg_d)
        return bool(icharg in [1, 11])

    def _check_chgcar(self, remote_folder):  # pylint: disable=no-self-use
        """
        Check if the CHGCAR file is present in the remote folder.

        This is only a very rudimentary test, e.g. we only check the
        presence of a file, not if its content is valid.
        """

        return 'CHGCAR' in remote_folder.listdir()

    def _check_wavecar(self, remote_folder):  # pylint: disable=no-self-use
        """
        Check if the WAVECAR file is present in the remote folder.

        This is only a very rudimentary test, e.g. we only check the
        presence of a file, not if its content is valid.
        """

        return 'WAVECAR' in remote_folder.listdir()

    def _need_wavecar(self):
        """
        Test wether a wavefunctions input is needed or not.

        :return output:
            True if a wavecar file must be
            used (py:method::NscfCalculation.use_wavefunctions),
            False otherwise
        needs 'parameters' input to be set
        (py:method::NscfCalculation.use_parameters)
        """
        istrt_d = 1 if self.inputs.get('wavefunctions') else 0
        istart = self._parameters.get('istart', istrt_d)
        return bool(istart in [1, 2, 3])

    def _structure(self):
        """
        Get the input structure as AiiDa StructureData.

        This is required in order to support CifData as input as well.
        """
        structure = self.inputs.structure
        if not hasattr(structure, 'get_pymatgen'):
            structure = get_data_node('structure', ase=structure.get_ase())
        return structure

    def write_additional(self, tempfolder, calcinfo):
        """Write CHGAR and WAVECAR files if needed."""
        super(VaspCalculation, self).write_additional(tempfolder, calcinfo)
        if self._need_chgcar():
            # If we restart, we do not require inputs, but we should have a basic check
            # that the CHGCAR file is present
            if not self._is_restart():
                chgcar = tempfolder.get_abs_path('CHGCAR')
                self.write_chgcar(chgcar, calcinfo)
            else:
                remote_folder = self.inputs.restart_folder
                if not self._check_chgcar(remote_folder):
                    raise FileNotFoundError('Could not find CHGCAR in {}'.format(remote_folder.get_remote_path()))
        if self._need_wavecar():
            # If we restart, we do not require inputs, but we should have a basic check
            # that the WAVECAR file is present
            if not self._is_restart():
                wavecar = tempfolder.get_abs_path('WAVECAR')
                self.write_wavecar(wavecar, calcinfo)
            else:
                remote_folder = self.inputs.restart_folder
                if not self._check_wavecar(remote_folder):
                    raise FileNotFoundError('Could not find WAVECAR in {}'.format(remote_folder.get_remote_path()))

    def write_incar(self, dst):  # pylint: disable=unused-argument
        """
        Write the INCAR.

        Passes the parameters node (Dict) from to the INCAR parser for
        preparation and writes to dst.

        :param dst: absolute path of the file to write to
        """
        incar_parser = IncarParser(data=self.inputs.parameters)
        incar_parser.write(dst)

    def write_poscar(self, dst):  # pylint: disable=unused-argument
        """
        Write the POSCAR.

        Passes the structures node (StructureData) to the POSCAR parser for
        preparation and writes to dst.

        :param dst: absolute path of the file to write to
        """
        settings = self.inputs.get('settings')
        settings = settings.get_dict() if settings else {}
        poscar_precision = settings.get('poscar_precision', 10)
        poscar_parser = PoscarParser(data=self._structure(), precision=poscar_precision)
        poscar_parser.write(dst)

    def write_potcar(self, dst):
        """
        Concatenates multiple POTCAR files into one in the same order as the elements appear in POSCAR.

        :param dst: absolute path of the file to write to
        """
        structure = self._structure()
        multi_potcar = MultiPotcarIo.from_structure(structure, self.inputs.potential)
        multi_potcar.write(dst)

    def write_kpoints(self, dst):  # pylint: disable=unused-argument
        """
        Write the KPOINTS.

        Passes the kpoints node (KpointsData) to the KPOINTS parser for
        preparation and writes to dst.

        :param dst: absolute path of the file to write to
        """
        kpoint_parser = KpointsParser(data=self.inputs.kpoints)
        kpoint_parser.write(dst)

    def write_chgcar(self, dst, calcinfo):  # pylint: disable=unused-argument
        charge_density = self.inputs.charge_density
        calcinfo.local_copy_list.append((charge_density.uuid, charge_density.filename, dst))

    def write_wavecar(self, dst, calcinfo):  # pylint: disable=unused-argument
        wave_functions = self.inputs.wavefunctions
        calcinfo.local_copy_list.append((wave_functions.uuid, wave_functions.filename, dst))

    @classmethod
    def _immigrant_add_inputs(cls, transport, remote_path, sandbox_path, builder, **kwargs):
        from aiida_vasp.calcs.immigrant import get_chgcar_input, get_wavecar_input  # pylint: disable=import-outside-toplevel
        add_wavecar = kwargs.get('use_wavecar') or bool(builder.parameters.get_dict().get('istart', 0))
        add_chgcar = kwargs.get('use_chgcar') or builder.parameters.get_dict().get('icharg', -1) in [1, 11]
        if add_chgcar:
            transport.get(remote_path.join('CHGCAR').strpath, sandbox_path.strpath)
            builder.charge_density = get_chgcar_input(sandbox_path)
        if add_wavecar:
            transport.get(remote_path.join('WAVECAR').strpath, sandbox_path.strpath)
            builder.wavefunctions = get_wavecar_input(sandbox_path)


def ordered_unique_list(in_list):
    """List unique elements in input list, in order of first occurrence."""
    out_list = []
    for i in in_list:
        if i not in out_list:
            out_list.append(i)
    return out_list
