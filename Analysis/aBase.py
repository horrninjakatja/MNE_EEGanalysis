##### Basic EEG Analysis with mne-python #####

import mne 

###Load and inspect the data###

raw = mne.io.read_raw_brainvision('../ExampleData/sem1a.vhdr')
print(raw)
print(raw.info)
raw.plot()
data,times = raw[:,:]
print(data.shape)

###Read events###

events = mne.find_events(raw, stim_channel='STI 014')

#rearrange, as "32" marks stimulus onset and previous trigger marks condition
for i,a in enumerate(events): 
    if events[i][2]==32:
        events[i-1][0]=events[i][0]

#cut
event_id = dict(target=1,standard=2,novelty=3) #Trigger Stimulus Onset
t_min = -0.2  #Prestimulus timespan
t_max =  1.0  #Poststimulus timespan
picks = mne.pick_types(raw.info,eeg=True,stim=False,exclude=['IO1'])
epochs = mne.Epochs(raw,events,event_id,t_min,t_max,picks=picks)
print(epochs)

###Evoked response###

evokedtarget = epochs['target'].average()
print(evokedtarget)
evokedtarget.plot()

evokedtarget = epochs['novelty'].average()
print(evokedtarget)
evokedtarget.plot()

evokedtarget = epochs['standard'].average()
print(evokedtarget)
evokedtarget.plot()