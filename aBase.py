##### Basic EEG Analysis with mne-python #####

import mne 
from epochdef import epochdef
from mne.preprocessing import ica

### Load and inspect the data ##############################################
############################################################################

raw = mne.io.read_raw_brainvision('../ExampleData/sem1a.vhdr')
raw.load_data()
#print(raw),print(raw.info),raw.plot()
#data,times = raw[:,:]
#print(data.shape)
chans = raw.ch_names[:]
chans.remove('IO1')
raw.pick_channels(chans)
chans.remove('STI 014')
mon = mne.channels.read_montage(kind='standard_1005',ch_names=chans)
#mne.viz.plot_montage(mon)

### Preprocessing ##########################################################
############################################################################

raw.set_eeg_reference('average', projection=True)
raw.filter(1,40)

### Read events and cut epochs #############################################
############################################################################

epochs = epochdef(raw)
epochs.set_montage(mon)
epochs.load_data()
print(epochs)

### Artefact correction ####################################################
############################################################################

epochs.plot(n_epochs=1,n_channels=63,block=True) #inspect & reject manually
#bad trials dropped, bad channels marked for interpolation after ICA

picks_ica=mne.pick_types(epochs.info,eeg=True,exclude='bads') #ica (excluding bad channels)
icadat = ica.run_ica(epochs,60,max_pca_components=62,random_state=40,picks=picks_ica)
icadat.plot_components() # make sure blink component is rejected
icadat.apply(epochs)

epochs.interpolate_bads(reset_bads=True) #interpolate channels
epochs.plot(n_epochs=1,n_channels=63,block=True)
reject=dict(eeg=5e-4) #reject based on histogram data
epochs.drop_bad(reject=reject)

#save clean data (and how to load for later use)
epochs.save('su01_clean-epo.fif')
#epochs = mne.read_epochs('su01_clean-epo.fif')

### Evoked response ########################################################
############################################################################

evokedtarget = epochs['target'].average()
evokednovelty = epochs['novelty'].average()
evokedstandard = epochs['standard'].average()

evokeds = {}
evokeds['1target']=evokedtarget
evokeds['2novelty']=evokednovelty
evokeds['3standard']=evokedstandard
mne.viz.plot_evoked_topo([evokedtarget,evokednovelty,evokedstandard])
mne.viz.plot_compare_evokeds(evokeds,gfp=False)

evokedtarget.plot_topomap(vmin=-10,vmax=10)
evokedstandard.plot_topomap(vmin=-10,vmax=10)
evokednovelty.plot_topomap(vmin=-10,vmax=10)


