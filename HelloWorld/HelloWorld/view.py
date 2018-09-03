from django.http import HttpResponse
from django.shortcuts import render

def hello(request):
	context		={}
	context['hello']="HHHH"
	context['p']	="p"
	return render(request,'hello.html',context)
