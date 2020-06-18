ssh ubuntu@172.18.140.183   pw:ubuntu


1. cd ~/EasyDarwin-linux-8.1.0-1901141151

2. start service
	sudo sh ./start.sh
	(if shows error: service exist, then remove the exist service and start again: sudo rm /etc/systemd/system/EasyDarwin_Service.service)

3. push your video file to easyDarwin server, (replace ../face.mp4 to your target video file)
	ffmpeg -re -i ../face.mp4 -vcodec copy -codec copy -f rtsp rtsp://172.18.140.183:554/test 

4. Open page administrator: 
	http://172.18.140.183:10008/login.html
	usr/pw: admin/admin

5. Pull the stream and watch in vlc:
	open vlc ——> media ——> open network stream ——> enter "rtsp://172.18.140.183:554/test"

6. Stop easyDarwin:
	sudo service EasyDarwin_Service stop
