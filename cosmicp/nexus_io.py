import h5py

"""
    This file serves the purpose to provide a simple dictionary that mimics the NeXus structure of the
    NXptycho defintion as described in the NXptycho.nxdl.xml file
    This originated to start off a simple integration of writing NXptycho NeXus structure at the
    COSMIC Imaging beamline at ALS after the frames underwent pre-processing.
"""

nexus_groups = {
    "instrument": "entry/instrument/",
    "beam": "entry/instrument/beam/",
    "detector": "entry/instrument/detector/",
    "detector_transformations": "entry/instrument/detector/transformations/",
    "sample": "entry/sample/",
    "sample_transformations": "entry/instrument/sample/transformations/"
}

#What is optional and mandatory? We could add a flag for that in each dataset.

nexus_data = {
    "data": nexus_groups["detector"] + "data",
}

nexus_metadata = {
    "definition": "entry/definition",
    "experiment_description": "entry/experiment_description",
    "title": "entry/title",
    # instrument fields
    "instrument_name": nexus_groups["instrument"] + "instrument_name",
    # beam fields
    "energy": nexus_groups["beam"] + "incident_beam_energy",
    # detector fields
    "x_pixel_size": nexus_groups["detector"] + "x_pixel_size",
    "y_pixel_size": nexus_groups["detector"] + "y_pixel_size",
    "detector_distance": nexus_groups["detector"] + "distance",
    "x_detector_translation": nexus_groups["detector_transformations"] + "x_translation",
    "y_detector_translation": nexus_groups["detector_transformations"] + "y_translation",
    "z_detector_translation": nexus_groups["detector_transformations"] + "z_translation",
    # sample fields
    "positioner_1_value": nexus_groups["sample"] + "positioner_1/raw_value",
    "positioner_1_name": nexus_groups["sample"] + "positioner_1/name",
    "positioner_2_value": nexus_groups["sample"] + "positioner_2/raw_value",
    "positioner_2_name": nexus_groups["sample"] + "positioner_2/name",
    "x_translations": nexus_groups["sample_transformations"] + "x_coarse_translation",
    "y_translations": nexus_groups["sample_transformations"] + "y_coarse_translation",
    "z_translations": nexus_groups["sample_transformations"] + "z_coarse_translation",
    "x_fine_translations": nexus_groups["sample_transformations"] + "x_fine_translation",
    "y_fine_translations": nexus_groups["sample_transformations"] + "y_fine_translation",
    "z_fine_translations": nexus_groups["sample_transformations"] + "z_fine_translation",
    "alpha_rotation": nexus_groups["sample_transformations"] + "alpha_rotation",
    "beta_rotation": nexus_groups["sample_transformations"] + "beta_rotation",
    "gamma_rotation": nexus_groups["sample_transformations"] + "gamma_rotation"

}

cosmic_metadata = {
    "illumination": "entry/probe",
    "illumination_mask": "entry/probe_mask",
}

def read(file_name, data_format = None, data_indexes = ()):

    if data_format is None: 
        data_format = nexus_metadata

    data_dictionary = {}
    try:
        with h5py.File(file_name, "r") as f:
            for key, value in data_format.items():
                if i in f : #if group exists...
                    data = f[data_indexes]
                    data_dictionary.update({key : data})
                    break
        return data_dictionary
    except IOError:
        print("ERROR opening file: " + file_name)
        return 0 

def write(file_name, data_dictionary, data_format = None):

    if data_format is None: 
        data_format = nexus_metadata

    with h5py.File(file_name, 'a') as f:

        if not "entry" in f: 
            f.create_group("entry")
    
        #data_format is used to generate hdf5 fields 
        for key, value in data_dictionary.items():
            #we may have here fields that are not considered in the formats above, we still handle and save them
            try:
                group = data_format[key]
            except KeyError:
                group = key + "/"
            if value is not None:
                f.create_dataset(group, data = value)

