from indica.examples import example_plasma


class TestPlasmaCaching:
    def setup_method(self):
        self.plasma = example_plasma()
        self.property_getters = [
            "Fz",
            "Ion_density",
            "Lz_tot",
            "Meanz",
            "Zeff",
            "Total_radiation",
            "Electron_pressure",
            "Thermal_pressure",
        ]
        self.properties = [
            "fz",
            "ion_density",
            "lz_tot",
            "meanz",
            "zeff",
            "total_radiation",
            "electron_pressure",
            "thermal_pressure",
        ]

    def test_caching_is_used_when_dependencies_dont_change(self):

        ion_density = self.plasma.ion_density
        assert (ion_density == self.plasma.ion_density).all()

    def test_caching_is_not_used_when_dependencies_change(self):

        ion_density = self.plasma.ion_density
        self.plasma.electron_density *= 0.5
        assert (ion_density != self.plasma.ion_density).any()

    def test_hash_changes_when_dependencies_change(self):

        _hash = self.plasma.Fz.__hash__()
        self.plasma.electron_temperature *= 2
        self.plasma.electron_density *= 2
        assert _hash != self.plasma.Fz.__hash__()

    def test_property_values_match_operator_calls(self):

        for property in self.properties:
            if (
                type(getattr(self.plasma, property)) is not dict
            ):  # fz should really not be a dict!
                assert (
                    getattr(self.plasma, property)
                    == getattr(self.plasma, f"calc_{property}")()
                ).all()

    def test_all_hashes_changes_when_dependencies_change(self):
        _hashes = {}
        for getter in self.property_getters:
            _hashes[getter] = getattr(self.plasma, getter).__hash__()

        self.plasma.electron_temperature *= 2
        self.plasma.electron_density *= 2

        for getter in self.property_getters:
            assert _hashes[getter] != getattr(self.plasma, getter).__hash__()
