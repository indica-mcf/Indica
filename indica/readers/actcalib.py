from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess
import sys
import warnings

import git

from indica.abstractio import BaseIO
from indica.readers import UDAUtils
from indica.utilities import CACHE_DIR


class VersionConflictError(RuntimeError):
    pass


class ACTVersionManager:
    def __init__(
        self,
        repo_url: str,
        repo_dir: str | Path,
    ):
        self.repo_url = repo_url
        self.repo_dir = Path(repo_dir).expanduser()

    def ensure_version(self, version: str):
        """
        Ensure that the requested revision is the one that
        will be imported.

        Returns the imported package.
        """

        # Determine the package name of requested version.
        # Assume version name  has form e.g. v1.0.0
        try:
            major_version = int(version.split(".")[0][1:])
        except ValueError:
            raise RuntimeError(
                f"The scheduler log recorded an unexpected ACT git tag: {version}"
            )
        if major_version < 5:
            package_name = "pyact"
            warnings.warn(
                (
                    f"Major version = {major_version} is not expected to work "
                    + "because it relies on an old version of pyaml which is "
                    + "incompatible with INDICA. Please consider analysing a more "
                    + "recent pass number."
                ),
                stacklevel=2,
            )
        else:
            package_name = "act"

        # Get the sha of the requested version
        expected_sha = self._resolve_sha(version)

        # Check for existing import of either act/pyact. If existing import exists,
        # raise an error if the files checked out in gitlab don't match the expected
        # version, otherwise return the module
        for _package_name in ["act", "pyact"]:
            if _package_name in sys.modules:
                current_sha = self._imported_package_sha(_package_name)
                if current_sha != expected_sha:
                    raise VersionConflictError(
                        f"The version of act currently imported into Python does "
                        f"not match the version which ran in the scheduler to make "
                        f"the shotnum and revision of act which has been requested. "
                        f"This is probably because this instance of Python has "
                        f"already been used to analyse a different MAST-U "
                        f"shotnum/passnum. To proceed with this anaylsis CGB "
                        f"suggests to start a new instance of Python\n\n"
                        f"{_package_name!r} already imported.\n"
                        f"Requested : {version} ({expected_sha[:8]})\n"
                        f"Imported  : {current_sha[:8]}\n"
                    )

                # Package is already installed and repo matches requested version...
                print(f"Using act version {current_sha}...")
                return sys.modules[package_name]

        # Given that no package has yet been in imported, checkout the required
        # version of the git repository
        self._checkout(expected_sha)

        # Import the package
        print(f"Importing {package_name}...")
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        package = import_module(package_name)

        return package_name, package

    def _repo(self) -> git.Repo:

        if not self.repo_dir.exists():
            print("Cloning act code repository...")
            repo = git.Repo.clone_from(
                self.repo_url,
                self.repo_dir,
            )
            print("Installing act code...")
            self._install_package()
            return repo

        return git.Repo(self.repo_dir)

    def _resolve_sha(self, revision: str) -> str:

        repo = self._repo()
        repo.remotes.origin.fetch(tags=True)

        return repo.commit(revision).hexsha

    def _checkout(self, sha: str):

        repo = self._repo()
        if repo.head.commit.hexsha != sha:
            print(f"Checking out act version {sha}...")
            repo.git.checkout(sha)

    def _imported_package_sha(self, package_name) -> str:

        package = sys.modules[package_name]
        package_root = Path(package.__file__).resolve()
        repo = git.Repo(package_root, search_parent_directories=True)

        return repo.head.commit.hexsha

    def _install_package(self):
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    str(self.repo_dir),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CallProcessError as e:
            raise RuntimeError(f"Failed to install package:\n{e.stderr}") from e


class ACTCalib(BaseIO):
    """ """

    def __init__(self, pulse):
        """Initialise class by pointing at correct repository"""

        # Set up repositories
        self._pulse = pulse
        self.set_act_calib_repository()
        self.set_act_repository()
        self.uda_utils = UDAUtils(pulse)

    def set_act_calib_repository(self):
        """Clone act_calib if necessary"""

        repo_dir = Path.home() / CACHE_DIR / "act_calib"
        repo_url = "git@gitlab.ukaea.uk:MAST-U_Scheduler/act_calib.git"
        if not repo_dir.exists():
            print("Cloning act calibration repository...")
            self._repo = git.Repo.clone_from(repo_url, repo_dir)
        else:
            self._repo = git.Repo(repo_dir)
        self._repo.remotes.origin.fetch(tags=True)

    def set_act_repository(self):
        """Clone act code if necessary and create instance of package manager
        pointing at code
        """

        # Update act repository to required version
        self.act_manager = ACTVersionManager(
            repo_url="git@gitlab.ukaea.uk:MAST-U_Scheduler/act.git",
            repo_dir=Path.home() / CACHE_DIR / "act",
        )

    def ensure_act_version(self, uid, instrument, revision):
        """Make sure that no other versions of act have been imported but the
        correct one
        """

        required_act_version = self.get_pulse_tag(
            uid,
            instrument,
            revision,
            repo="code",
        )
        return self.act_manager.ensure_version(required_act_version)

    def make_virtualmeter(self, uid, instrument, revision):
        """Make a virtualmeter using the act code"""

        # Set up repositories
        self.checkout_scheduler_act_calib_tag(uid, instrument, revision)
        package_name, _ = self.ensure_act_version(uid, instrument, revision)

        # Import the required functions
        runact = import_module(f"{package_name}.run{package_name}")
        run_act = getattr(runact, f"run_{package_name}")
        set_up_runinfo = getattr(runact, "set_up_runinfo")

        # Get run card
        tag = self.uda_utils._mastu_names(instrument)
        runinfo = set_up_runinfo(
            self._pulse,
            tag,
            calibration_directory=self._repo.working_tree_dir,
        )

        # Construct virtualmeter
        if instrument in runinfo.run_card["Virtualmeters"].keys():

            # Remove reference views
            # Fitting refernce views crashes the code and is not strictly necessary
            vm_info = runinfo.run_card["Virtualmeters"][instrument]
            sd_opts = vm_info["VirtualmeterDataOptions"]["spectrometerdata_options"]
            for spectrometer, options in sd_opts.items():
                options["reference_views"] = []

            # Extract relevant virtualmeter options
            virtualmeters = {instrument: runinfo.run_card["Virtualmeters"][instrument]}

            # Make virtualmeter
            output = run_act(runinfo, execute=False, virtualmeters=virtualmeters)

        return output["Virtualmeters"][instrument]

    def checkout_scheduler_act_calib_tag(self, uid, instrument, revision):
        """Checkout the relevant act_calib git commit"""
        checked_out_tags = self.get_checked_out_act_calib_tags()
        pulse_tag = self.get_pulse_tag(uid, instrument, revision)
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

    def get_checked_out_act_calib_tags(self):
        """Deduce which commit is checked out"""
        head_commit = self._repo.head.commit
        tags = [tag.name for tag in self._repo.tags if tag.commit == head_commit]
        return tags

    def get_pulse_tag(self, uid, instrument, revision, repo="calib"):
        """Find which commit was used to calculate scheduler results.

        repo must be either "calib" or "code"

        """
        # Set up the UDA stuff
        tag = self.uda_utils._mastu_names(instrument)
        passnum = self.uda_utils.get_revision(uid, instrument, revision)[0]

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

    def close(self):
        pass

    def requires_authentication(self):
        pass
