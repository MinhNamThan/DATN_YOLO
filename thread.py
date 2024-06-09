from threading import Thread
import threading
import time

def cal_square(numbers):
	print("calculate square number")
	for n in numbers:
		time.sleep(0.2)
		print ('square:', n*n)


def cal_cube(numbers):
	print("calculate cube number \n")
	for n in numbers:
		time.sleep(0.2)
		print ('cube:', n*n*n)

arr = [2,3,7,9]
pIDs = []

def delete(url):
	for pid in pIDs:
		if url == pid.get("url"):
			os.kill(pid.get("pid"))


def a(url)
	t = time.time()
	t1 = threading.Thread(target=cal_square, args=(url,))
	t1.start()
	pIDs.append({"pid":str(t1.pid)}, "url": url)

	t1.join()
