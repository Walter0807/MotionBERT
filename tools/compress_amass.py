import numpy as np
import os
import pickle

raw_dir = './data/AMASS/amass_202203/'
processed_dir = './data/AMASS/amass_fps60'
os.makedirs(processed_dir, exist_ok=True)

files = []
length = 0
target_fps = 60

def traverse(f):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        # file
        if not os.path.isdir(tmp_path):
            files.append(tmp_path)
        # dir
        else:
            traverse(tmp_path)

traverse(raw_dir)

print('files:', len(files))

fnames = []
all_motions = []

with open('data/AMASS/fps.csv', 'w') as f:
    print('fname_new, len_ori, fps, len_new', file=f)
    for fname in sorted(files):
        try:
            raw_x = np.load(fname)
            x = dict(raw_x)
            fps = x['mocap_framerate']
            len_ori = len(x['trans'])
            sample_stride = round(fps / target_fps)
            x['mocap_framerate'] = target_fps
            x['trans'] = x['trans'][::sample_stride]
            x['dmpls'] = x['dmpls'][::sample_stride]
            x['poses'] = x['poses'][::sample_stride]
            fname_new = '_'.join(fname.split('/')[2:])
            len_new = len(x['trans'])
            
            length += len_new
            print(fname_new, ',', len_ori, ',', fps, ',', len_new, file=f)
            fnames.append(fname_new)
            all_motions.append(x)
            np.savez('%s/%s' % (processed_dir, fname_new), x)
        except:
            pass
        
#         break

print('poseFrame:', length)
print('motions:', len(fnames))

with open("data/AMASS/all_motions_fps%d.pkl" % target_fps, "wb") as myprofile:  
    pickle.dump(all_motions, myprofile)
    
