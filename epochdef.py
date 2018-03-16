##### Define trials and cut #####

import mne

def epochdef(raw):
    events = mne.find_events(raw, stim_channel='STI 014')
    
    #adapt to find trigger of interest
    #here: "32" marks stimulus onset and previous trigger marks condition
    for i,a in enumerate(events): 
        if events[i][2]==32:
            events[i-1][0]=events[i][0]
            
    #cut

    event_id = dict(target=1,standard=2,novelty=3) #Trigger Stimulus Onset
    t_min = -0.2 #Prestimulus timespan
    t_max =  1.0 #Posttimulus timespan
    picks = mne.pick_types(raw.info,eeg=True,stim=False)
    epochs = mne.Epochs(raw,events,event_id,t_min,t_max,picks=picks)
    return(epochs)


