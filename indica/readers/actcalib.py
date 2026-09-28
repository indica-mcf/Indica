from __future__ import annotations

from pathlib import Path
import pickle
import stat
import warnings

from act.loadcals import CalData
from act.runact import run_act
from act.runact import set_up_runinfo
import git
import numpy as np

from indica.abstractio import BaseIO
from indica.readers import UDAUtils
from indica.utilities import CACHE_DIR
from indica.utilities import read_nested_dict
from indica.utilities import to_filename

KERNEL_HALFPIX = 25


class ACTCalibError(Exception):
    """An exception which occurs when trying to read act_calib data which would
    not be otherwise caught by pyuda

    """


class ACTCalibWarning(UserWarning):
    """A warning that occurs while trying to read act calib data."""


class VersionConflictError(RuntimeError):
    pass


class ACTCalib(BaseIO):
    """THIS class is a temporary solution until the ACT intershot code outputs
    complete information to UDA through the scheduler.

    Because it is temporary, some shortcuts have been taken.

    The class is only suitable to be used using the latest version of the ACT
    code and the latest version of the calibratino repository

    """

    # _loadcals = None
    # _runact = None

    # @property
    # def loadcals(self):
    #    if self._loadcals is None:
    #        import act.loadcals as loadcals
    #        self.__class__._loadcals = loadcals
    #    return self._loadcals

    # @property
    # def runact(self):
    #    if self._runact is None:
    #        import act.runact as runact
    #        self.__class__._runact = runact
    #    return self._runact

    def __init__(self, pulse, kernel_halfpix=25):
        """Initialise class by pointing at correct repository"""

        # Set up repositories
        self._pulse = pulse
        self._reader_cache_id = "mastu_virtualmeters"
        self._set_act_calib_repository()
        self.uda_utils = UDAUtils(pulse)
        self.kernel_halfpix = kernel_halfpix

    def _set_act_calib_repository(self):
        """Clone act_calib if necessary"""

        repo_dir = Path.home() / CACHE_DIR / "act_calib"
        repo_url = "git@gitlab.ukaea.uk:MAST-U_Scheduler/act_calib.git"
        if not repo_dir.exists():
            print("Cloning act calibration repository...")
            self._repo = git.Repo.clone_from(repo_url, repo_dir)
        else:
            self._repo = git.Repo(repo_dir)
        self._repo.remotes.origin.fetch(tags=True)

    def get_vm_data(self, data_path, instrument, revision):
        """Retrieve data from virtualmeter"""

        self.check_revision(instrument, revision)
        vm = self._get_virtualmeter(instrument, revision)
        data = read_nested_dict(vm.data, data_path.split("/"))
        return np.transpose(data)

    def check_revision(self, instrument, revision):
        """Check that the revision requested is the latest one"""

        latest_passnum, is_best = self.uda_utils.get_revision(
            "UDADefault",
            instrument,
            revision,
        )
        if not is_best:
            raise ACTCalibError(
                "ACTCalib is only suitable to use with the latest passnumber. "
                + f"Passnum {latest_passnum} has been requested, which is not the "
                + f"latest pass for shot {self._pulse}"
            )

        return latest_passnum

    def _get_virtualmeter(self, instrument, revision):
        """Load virtualmeter from cache, or otherwise construct it"""

        # Attempt to read from cache
        latest_passnum = self.check_revision(instrument, revision)
        cache_path = self._vm_args_to_file(instrument, latest_passnum)
        vm = self._read_cached_vm_file(cache_path)

        if vm is None:
            try:
                vm = self._make_virtualmeter(instrument)
            except Exception as e:
                raise ACTCalibError(
                    "ACTCalib could not construct virtualmeter for "
                    + f"{instrument} revision {revision} and pulse {self._pulse}"
                ) from e
            self._write_cached_uda_file(cache_path, vm)

        return vm

    def _vm_args_to_file(self, instrument, passnum):
        """Determine file path for caching virtualmeter"""

        id_list = [self._reader_cache_id, self._pulse, instrument, passnum]
        id_list = [str(x) for x in id_list]
        return (
            Path.home()
            / CACHE_DIR
            / self.__class__.__name__
            / to_filename("_".join(id_list) + ".pkl")
        )

    def _read_cached_vm_file(self, path: Path):
        """Read a cached virtualmeter from a file"""

        if not path.exists():
            return None
        permissions = stat.filemode(path.stat().st_mode)
        if permissions[5] == "w" or permissions[8] == "w":
            warnings.warn(
                "Can not open cache file which is writeable by anyone other than "
                "the user. (Security risk.)",
                ACTCalibWarning,
            )
            return None
        with path.open("rb") as f:
            try:
                print(f"Reading {path}")
                vm = pickle.load(f)
                return vm
            except pickle.UnpicklingError:
                warnings.warn(
                    f"Error unpickling cache file {path}. (Possible data corruption.)",
                    ACTCalibWarning,
                )
                return None

    def _write_cached_uda_file(self, path, virtualmeter):
        """Write the given virtualmeter, compiled by make vm, to the disk for
        later resuse
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(virtualmeter, f)
        path.chmod(0o644)

    def _make_virtualmeter(self, instrument):
        """Make a virtualmeter using the act code"""

        # Get run card
        tag = self.uda_utils._mastu_names(instrument).split("/")[0]
        runinfo = set_up_runinfo(
            self._pulse,
            tag,
            calibration_directory=self._repo.working_tree_dir,
        )

        # Construct virtualmeter
        if instrument not in runinfo.run_card["Virtualmeters"].keys():
            raise ACTCalibError(
                f"Cannot find calibration record for {instrument} for {self.pulse}"
            )

        # Remove reference views because fitting refernce views crashes the code
        # and is not strictly necessary
        vm_info = runinfo.run_card["Virtualmeters"][instrument]
        sd_opts = vm_info["VirtualmeterDataOptions"]["spectrometerdata_options"]
        for spectrometer, options in sd_opts.items():
            options["reference_views"] = []

        # Extract relevant virtualmeter options
        virtualmeters = {instrument: runinfo.run_card["Virtualmeters"][instrument]}

        # Make virtualmeter
        output = run_act(runinfo, execute=False, virtualmeters=virtualmeters)
        vm = output["Virtualmeters"][instrument]

        # Load the pupils
        runinfo = output["RunInfo"]
        pupils = CalData("common/effective_pupils", runinfo).data

        # Add sightline directions
        dimlist = ["X", "Y", "Z"]
        for view_name, view_data in vm.data.items():
            view_data["Location"] = pupils[view_name]["pupil_pos"]
            direction = []
            for index, spectrometer_fibre in enumerate(view_data["SpectrometerFibre"]):
                spectrometer = view_data["Spectrometer"][index]
                qs = "SpectrometerFibre==@spectrometer_fibre"
                row = vm.spectrometerdata[spectrometer].config.query(qs).iloc[0]
                channel_direction = [None] * 3
                for index, dim in enumerate(dimlist):
                    channel_direction[index] = row[f"VslScr{dim}"]
                direction.append(channel_direction)
            view_data.update({"Direction": direction})

        # Add instrument functions
        for view_name, view_data in vm.data.items():
            inst_func = np.full(
                ((2 * self.kernel_halfpix) + 1, view_data["NumChords"]), np.nan
            )
            for index, spectrometer_fibre in enumerate(view_data["SpectrometerFibre"]):
                inst_func_model = view_data["InstFuncModel"][index]
                inst_func_params = view_data["InstFuncParams"][index]
                w = view_data["Wavelength"][:, index]
                dx = w[1] - w[0]
                inst_func_x = np.linspace(
                    -dx * self.kernel_halfpix,
                    dx * self.kernel_halfpix,
                    1 + (2 * self.kernel_halfpix),
                )
                inst_func[:, index] = inst_func_model(
                    inst_func_x,
                    0,
                    inst_func_params,
                )
            view_data["InstrumentFunction"] = inst_func

        return vm

    def _checkout_scheduler_act_calib_tag(self, uid, instrument, revision):
        """#####  UNUSED  ##### Checkout the relevant act_calib git commit"""
        checked_out_tags = self._get_checked_out_act_calib_tags()
        pulse_tag = self._get_pulse_tag(uid, instrument, revision)
        if pulse_tag not in checked_out_tags:
            try:
                self._repo.git.checkout(pulse_tag)
            except git.exc.GitCommandError as e:
                e.stderr = (
                    "\n\nERROR: SOME ACT_CALIB TAGS GOT LOST IN GITLAB MIGRATION\n"
                    + "\nProblem is likely caused by absence of tags in the remote "
                    + "git repository\n"
                    + "CGB has opened an Ivanti ticket (10/Sep/2026)\n"
                    + e.stderr
                    + "\n\n"
                )
                raise e

    def _get_checked_out_act_calib_tags(self):
        """#####  UNUSED  ##### Deduce which commit is checked out"""
        head_commit = self._repo.head.commit
        tags = [tag.name for tag in self._repo.tags if tag.commit == head_commit]
        return tags

    def _get_pulse_tag(self, uid, instrument, revision, repo="calib"):
        """#####  UNUSED  ##### Find which commit was used to calculate scheduler
        results. Repository must be either "calib" or "code"

        """
        # Set up the UDA stuff
        tag = self.uda_utils._mastu_names(instrument)
        passnum, is_best = self.uda_utils.get_revision(
            "UDADefault",
            instrument,
            revision,
        )
        if not is_best:
            raise RuntimeError("ACTCalib only supports use of the most recent UDA pass")

        # Determine some log strings
        if repo == "calib":
            section_string = "calib.git"
        elif repo == "code":
            section_string = "act.git"
        else:
            raise ValueError("Repo must be either `calib` or `code`")

        # Find git commit
        filepath = (
            f"$MAST_DATA/{self._pulse}/Pass{passnum}/" + f"{tag}_{self._pulse:06d}.log"
        )
        log = bytes(
            self.uda_utils._client.get(f"bytes::read(path={filepath})").data
        ).decode(encoding="utf-8")
        # print(log)
        section_pos = log.find(section_string)
        latest_pos = log[section_pos:].find("Latest git tag") + section_pos
        delim = "-----\n"
        version_pos = log[latest_pos:].find(delim) + latest_pos + len(delim)
        calib_git_tag = log[version_pos : log[version_pos:].find("\n") + version_pos]

        return calib_git_tag

    def _get_calib_file(self):
        pass

    def close(self):
        pass

    def requires_authentication(self):
        pass


def clone_act_repository():
    """Clone act code if necessary and create instance of package manager
    pointing at code
    """

    # Set up ACT
    repo_url = "git@gitlab.ukaea.uk:MAST-U_Scheduler/act.git"
    repo_dir = Path(Path.home() / CACHE_DIR / "act").expanduser()

    if not repo_dir.exists():
        print("Cloning act code repository...")
        repo = git.Repo.clone_from(
            repo_url,
            repo_dir,
        )
        repo.git.checkout("master")
        raise RuntimeError(
            "The MAST-U CXRS intershot code, `ACT` was not installed. This has "
            + f"now been cloned to {repo_dir}. Please install using pip, and then "
            "try again."
        )


if __name__ == "__main__":
    # try:
    #    from act.runact import set_up_runinfo, run_act
    #    from act.loadcals import CalData
    # except ModuleNotFoundError:
    #    clone_act_repository()
    act_calib = ACTCalib
