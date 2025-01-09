import os
import random
fp=open('IJ.csv','w')
mylink=[]

mylink1='https://redmist328.github.io/MOS/nog/I/'
mylink2='https://redmist328.github.io/MOS/nog/J/'
# mylink3='https://redmist328.github.io/MOS/mkdear/C/'
#mylink4='https://yxlu-0102.github.io/MOS/MOS_LJSpeech_Ablation/D/'ddd
# mylink5='https://yxlu-0102.github.io/MOS/MOS_LJSpeech_Ablation/E/'
# mylink6='https://yxlu-0102.github.io/MOS/MOS_LJSpeech_Ablation/F/'

mylink.append(mylink1)
mylink.append(mylink2)
# mylink.append(mylink3)
#mylink.append(mylink4)
# mylink.append(mylink5)
# mylink.append(mylink6)

for i in range(20):
	for j in range(2):
		fp.write('link'+str(i+i+j+1)+',')
fp.write('\n')

link_number_list=[0,1]

for i in range(20):
	if i < 9:
		name = '0'+str(i+1)+'.wav'
	else:
		name = str(i+1)+'.wav'

	random.shuffle(link_number_list)
	for j in range(2):
		fp.write(mylink[link_number_list[j]]+mylink[link_number_list[j]].split('/')[-2]+'_'+name+',')

fp.write('\n')
fp.close()