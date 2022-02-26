
#
# Only Gaia EDR3 and Gaia DR2 photometric systems are supported
cmd_systs = {
    'gaiaedr3': (
        ('Gmag', 'G_BPmag', 'G_RPmag'), (6390.21, 5182.58, 7825.08))
}
# line where the header starts in the CMD isochrone files
idx_header = 12
# Column names in isochrone files
logAge_name, label_name = 'logAge', 'label'
# Column names
mag_name, col_name, e_mag_name, e_col_name = 'Gmag', 'BP-RP', 'e_Gmag', 'e_BP-RP'
