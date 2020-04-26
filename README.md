# ParkingAnalytics

This directory holds all files I used while working on the project. 
The main idea was to create a single program that can do following: 
- Detect and mark all parking lots on the video 
- Detect and track all cars on the video 
- Identify the type of the car (minivan, truck etc.) 
- Detect and read Russian license plates on those cars 
- Store information about where each car has parked and how long it has been parked there 
### Details 
I didn't manage to create a single program but I've made several. Each of them completes a specific task from above. 
Best working versions of programs are 
• license_plate_recognition_v2.py 
• parking_lot_detection_v2.py 

Other programs I used for side tasks. For example “rotate.py” can be used to find the angle which the longest straight line on the photo has and to rotate that photo so that this line had and angle that we need (line 90 degrees for example). The other one “set_filter”.py is used to create a (as I call it) “color filter”. It allows you to highlight the color that you need on the photo. Imagine that you have a photo of a zebra and you need to only see white lines of her skin. So using this program you move the sliders to take away all other colors. This is used for parking lot detection as well. 
File “main.py” is unfinished. It doesn't work! Don't even try. I'll just leave it here… 
“plates_cascade.xml” contains data for detecting license plates (only Russian!!!) 
“parking_lots” folder contains some testing photos. Have a look at them. 
“license_plates” folder contains all cut fragments of photos that have enlarged license plates on them. Basically, it's used just to make sure everything works fine. 


To make “license_plate_recognition.py” work you will need to change some paths in the code. For example in the 9th line put the path to the photo of the car you would like to process. Line 12 contains the path to plates cascade. And the line 13 contains path to “tesseract.exe” app. This app should be installed separately by yourself! It's used for text recognition of the license plate. Also change line 262 and 264 to the paths where you would like to store a cut photo of the number. It will then be used to find text on it. 

After changing all paths just run the program. I hope it will work on your PC. Yes, I know that I should have used relative but not absolute paths. Pardon me for that. 

Have fun!
