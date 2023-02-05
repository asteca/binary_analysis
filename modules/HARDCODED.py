
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
mag_name, col_name, e_mag_name, e_col_name = 'Gmag', 'BP-RP', 'e_Gmag',\
    'e_BP-RP'

# Array for the q(=M2/M1) mass-ratio values. This determines how many
# isochrones will be stored in 'isoch_binaries'. A larger number means
# better resolution (at the expense of more memory used)
q_min, q_max, q_dist_N = 0.01, .99, 500

# Number of runs with a new IMF re-sample
N_IMF = 500
# Value used to decide which stars are binaries
bf_cut = .1

# Mass range for single stars
mmin, mmax = 0.08, 150

# gamma range used to find the gamma_q values that are used to sample the q
gamma_m_min, gamma_m_max = .1, 10
gamma_q_min, gamma_q_max = .1, 10
DE_tol = 0.001
