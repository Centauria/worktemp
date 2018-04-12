# -*- coding: utf-8 -*-

import re
import sys

if __name__=='__main__':
	if len(sys.argv)==2:
		target=sys.argv[1].upper()
		with open('kanji-pingyin.txt','r',encoding='UTF-8') as file_kanji:
			kanji_list=file_kanji.readlines()
		kanji_dict={}
		for line in kanji_list:
			k_v=line.split()
			kanji_dict[k_v[0]]=k_v[1]
		print('All kanji loaded. lines:',len(kanji_dict))
		with open('myouji.txt','r',encoding='UTF-8') as file_myouji:
			myouji_list=file_myouji.readlines()
		selected=[]
		for key in filter(lambda k:bool(re.search(target,k)),kanji_dict.keys()):
			for v in kanji_dict[key]:
				selected.append(v)
		result=filter(lambda m:selected.count(m)>0,map(lambda l:l.split()[0],myouji_list))
		print('Find:',list(result))
	else:
		print('1 pinyin at a time')