import os
import subprocess

#Create CSV with Header, comment out if header exists
csv_hdr = 'label,a,b,c,d,e,f,g,h,i,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y\n'
with open('testing_res.csv','a') as fd:
    fd.write(csv_hdr)


start_dir = 0
start_fil = 0
#Testing Images
cmd_str = 'python label_image.py --graph=baseline_tf_inception_v3_graph.pb --labels=baseline_tf_inception_v3_labels.txt --input_layer=Placeholder --output_layer=final_result --image=testing_images\\'

for curr_dir in os.listdir('testing_images')[start_dir:]:
    for curr_fil in os.listdir('testing_images\\' + curr_dir)[start_fil:]:
        res = subprocess.Popen(cmd_str + curr_dir + '\\'+ curr_fil,stdout=subprocess.PIPE)
        out = res.communicate()[0].decode("utf-8")
        with open('testing_res.csv','a') as fd:
            fd.write(curr_dir + out+'\n')
        res.kill()
    start_fil = 0 #only use start_fil for the first directory
        
