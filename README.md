# conseil_national

Data preparation:

The exact data that we used can be found at https://www.parlament.ch/de/ratsbetrieb/abstimmungen/abstimmungs-datenbank-nr, and by then choosing one of the three legislatures available and downloading the full data. The downloaded .csv files must be placed in the folder data.

A warning: The data that you download from the site contains missaligned columns that will most likely cause problems when trying to 
create the adjacency matrix.

The dataset containing all the councillor names with their party association is found in the data folder (Ratsmitglieder_1848_FR.csv)

Other requirements:

Three empyt folders in the Final_Project directory, called 'adjacencies','node_indices','figures'

The Louvain method for modularity calculations: can be installed with 'conda install python-louvain'
