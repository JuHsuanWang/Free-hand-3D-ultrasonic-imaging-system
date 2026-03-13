"""

[Program Name] PSRT_Python.py
[Description] implement user data proessing algorithm in real time under Python environment
[Usage]

    steps to operate PSRT :

    - step 1 : find the system.ini file in Prodigy installation folder, set "EnableRemoteControlPS" to 1
               at the beginning of this Python script, set the path and IP/port in 'user setting' section
   	- step 2 : at the "modify TX/RX parameters then start scan" section, modify parameters if required
   	- step 3 : run this Python script

   	then RF/beam-formed data will be sent from Prodigy to Python environment
   	user can process the data with their own algorithms within 'User_Application()'
    please check PSRT user manual for details of using PSRT

[Parameter Description]

                            RF data  line-by-line   ultrafast    photo-acoustic
	BeamFormingMethod          0           1           2              3
                            
                                
                      asynchronous mode     synchronous mode    one shot mode                    
    EnProcessDataType       0                   1                       2
    ScanStatus = -1                     before running any mode  
    ScanStatus =  0        stop                pause                  stop
    ScanStatus =  1        run                 run                    run
    ScanStatus =  2        N/A                 stop                   N/A
    
  
"""
# %% import necessary libraries
import sys
import time, timeit
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import threading
from prodigy import init_dll, Receive_data, Receive_data_cuda, socket_send_recv, socket_get_error, socket_wait_for_clear_error, format_value, Detected_NotMatch, get_parameters, start_ui, Check_PRF, update_Overall_para
import importlib.util
if importlib.util.find_spec("cupy") is not None:
    import cupy as cp
    print("   >> cp.__version__: ", cp.__version__)
    print("   >> cp.cuda.runtime.runtimeGetVersion(): ", cp.cuda.runtime.runtimeGetVersion())
else:
    print("CuPy is not installed.")

# Import our custom recorder module
from Core.recorder import DualPlaneRecorder

# Initialize the recorder, setting the desired output filenames
my_recorder = DualPlaneRecorder(left_filename="L_s.avi", right_filename="R_s.avi", fps=10)

# %% user setting

# --- path setting ---
parametersini_path = '../2.0/parameters.ini'
PSRT_dll_path = './'   # location of PSRT_Python.dll, set as './' if under the same folder of this Python script
para_file     = 'D:/NTU_Isaac_example/PS mode parameter file/B_mode_line_by_line_slot_duplicate - Eunice_implementation.ini'

# %% load and set-up PSRT_dll
CUDA_lib, PSRT_dll = init_dll(PSRT_dll_path)

# %% user application function definition

# --- An example of user applicatin for showing images from Prodigy system ---
# --- user can define their own applications here ---
photo_acoustic_lut = np.loadtxt('../2.0/hot_colormap_rgb.csv', delimiter=',', skiprows=1, dtype=np.uint8)

def User_Application(Overall_para, EG_para, EG_data):   

    for idx_evt in range(EventGroup_num):
        EG_data_host = EG_data[idx_evt]
        if (EG_para[0]['BeamFormingMethod'] == 0):     # ----- RF Data -----
    
            idx_ch  = 0
            idx_BeamRep = 0
            idx_Polarity = 0
            idx_Scanline = 0
            idx_GroupRepeat = 0
            data_tmp = np.squeeze(EG_data_host[:, idx_ch, idx_Polarity, idx_Scanline, idx_BeamRep, idx_GroupRepeat])
            plt.figure(idx_evt+1)
            plt.plot(data_tmp)
            plt.title('RF data')
            plt.xlabel('sample')
            plt.ylabel('ADC value')
            plt.grid(True)
            plt.show()
            plt.pause(0.1)
            
        else:
            
            if Overall_para['DataTypetoPlatform'] == 0:     # ----- BF Data -----
            
                B_Dr_Max = 0
                B_Dr_Min = -50
                
                if EG_para[idx_evt]['BeamFormingMethod'] == 1:  # line-by-line
                    ImageWidth = EG_para[idx_evt]['FOV']
                else:                                           # ultrafast or photo-acoustic
                    ImageWidth = Overall_para['Pitch']*Overall_para['Na']*1000
                ImageDepth = EG_para[idx_evt]['MaxDepth'] - EG_para[idx_evt]['MinDepth']
                    
                data_tmp = np.squeeze(EG_data_host)
                if len(data_tmp.shape) >= 3:
                    data_bf = data_tmp[:,:,0]
                else:
                    data_bf = data_tmp
                data_bf = data_bf.transpose()
                data_bf_abs = abs(data_bf)
                if data_bf_abs.max() == 0 :
                    data_bf_abs = np.ones(data_bf_abs.shape)
                else:    
                    data_bf_abs[np.where(data_bf_abs == 0)] = np.min(data_bf_abs[np.where(data_bf_abs > 0)])
                img_f_resize = Image.fromarray(20*np.log10(data_bf_abs))
            
                # --- linear array scan conversion ---
                if(ImageDepth > ImageWidth):
                    img_f_resize = np.array(img_f_resize.resize((512, round(ImageWidth / ImageDepth * 512))))
                else:
                    img_f_resize = np.array(img_f_resize.resize((round(ImageDepth / ImageWidth * 512), 512)))
                
                img = img_f_resize + EG_para[idx_evt]['GainOffset']           
                img[np.where(img < B_Dr_Min)] = B_Dr_Min
                img[np.where(img > B_Dr_Max)] = B_Dr_Max
                img = 255*(np.transpose(img) - B_Dr_Min)/(B_Dr_Max - B_Dr_Min)
                img = np.uint8(img)
        
            elif Overall_para['DataTypetoPlatform'] == 1:    # ----- Image -----

                img = np.squeeze(EG_data_host).astype('uint8')             

                # --- Split Data and Record ---
                if Overall_para['ProjectCode'] == 'A04' and Overall_para['EnSlotDuplicate'] == 1:
                    imgL = img[:,:,0]  # Extract Plane A (Left)
                    imgR = img[:,:,1]  # Extract Plane B (Right)
                    
                    # Call the recorder to save these two frames into .avi files
                    my_recorder.write_frames(imgL, imgR)

                if EG_para[idx_evt]['BeamFormingMethod'] == 3: # photo-acoustic
                    img = photo_acoustic_lut[img]
                    
            else:                                            # ----- Image + Data, to be implemented -----
                img = np.zeros((512,512))
        
            # --------------- plot images ---------------
            # (plt.imshow has been removed to prevent rendering lag during real-time receiving)
            # (Visualization will be handled by the main GUI later)
            
            # --- using open-cv
            #cv2.imshow(f'EG {idx_evt + 1}',img)
            #cv2.setWindowTitle(f'EG {idx_evt + 1}', 'after execution, press any key(instead of the "X" button) to close this window')
            #cv2.moveWindow('Image', 0, 0)  # Move it to (40,30)
            #cv2.waitKey(1)
               
#%% switch slot, transducer.ini file and modify TX/RX parameters then start scan
# user must confirm the existence of key to set TX/RX parameters
# ,setting on an incorrect key does not report error for now
#print('>> ----- switch slot/transducer.ini file or modify TX/RX parameters -----')
# --- switch slot (1 : 256 channel slot/ 2 or 3 : 128 channel slot)
# string_recv = socket_send_recv('<SET_SLOT>3')
# socket_wait_for_clear_error('Slot Setting is not completed.')
# string_recv = socket_send_recv('<SET_PROBE>Bor_Concave10M_8ch##General#4.22')
# socket_wait_for_clear_error('Probe Setting is not completed.')
            
# %% read default parameter values from parameter.ini file, and store to "Overall_para" and "EG_para"

print(f'\n>> ----- read default parameter values from {para_file.split("/")[-1]} -----')

# --- get parameter settings from "para_file"
Overall_para, EG_para = get_parameters(para_file)

# --- modify any TX/RX parameter for user's application here
Overall_para['EnProcessDataType'] = 2         # 0 : asynchronous mode / 1 : synchronous mode / 2 : one-shot mode
Overall_para['DataTypetoPlatform'] = 1        # 0: Data / 1: Image
# Overall_para['EnShowData'] = 0                # 0 : disable show beamform image / 1 : enable
Overall_para['Frames'] = 300  
# Overall_para['soundv'] = 1540
# Overall_para['ElementCoordinatePath'] = 'D:\SSUS Database\Demo Operator\PS\Rec2D_test.trc'
# Overall_para['FocusCoordinatePath'] =  'D:\SSUS Database\Demo Operator\PS\Rec2D_test.foc'
# Overall_para['EnStopWhenMemFull_Platform'] = 1         # 0 : disable stop memory full / 1 : enable

# for idx_evt in range(Overall_para['EventGroup_num']):
    # EG_para[idx_evt]['Freq'] = 6.4                    # TX frequency, MHz
    # EG_para[idx_evt]['Cycle'] = 2                     # TX cycle
    # EG_para[idx_evt]['BeamRep_UI'] = 1                  
    # EG_para[idx_evt]['MaxDepth'] = 50
    # EG_para[idx_evt]['PRF'] = 1
    # EG_para[idx_evt]['GainOffset'] = 20
    
    # EG_para[idx_evt]['Ext_Trig_In_En'] = 1           # 0: Disable trigger, 1: Enable trigger in
    # EG_para[idx_evt]['Ext_Trig_In_Edge'] = 0         # 0: Rising, 1: Falling
    # EG_para[idx_evt]['Ext_Trig_In_Dual_Edge'] = 0    # 0: Disable, 1: Dual Edge
    # EG_para[idx_evt]['Ext_Trig_In_Connector'] = 1    # 1: Connector#1, 2: Connector#2
    # EG_para[idx_evt]['Ext_Trig_In_Type'] = 0         # 0: Event trigger, 1: Frame trigger, 2: Event Group trigger
    
    # EG_para[idx_evt]['Ext_Trig_Out_Level_En'] = 0    # 0: Disable trigger, 1: Enable trigger in
    # EG_para[idx_evt]['Ext_Trig_Out_Level'] = 0       # 0: Rising, 1: Falling
    # EG_para[idx_evt]['Ext_Trig_Out_Connector'] = 1    # 1: Connector#1, 2: Connector#2
    # EG_para[idx_evt]['Ext_Trig_Out_Type'] = 0         # 0: Event trigger, 1: Frame trigger, 2: Event Group trigger
    
    # --- note : if Overall_para['DataTypetoPlatform'] = 1(image), do not set BeamFormingMethod to 0(RF data)
    # EG_para[idx_evt]['BeamFormingMethod'] = 0         # 0 : RF data/ 1 : line-by-line scan / 2 : ultrafast / 3 : photo-acoustic
                                                        

# --------------------- write parameters then start scan ----------------------
string_send = "[Overall]\r\n"
for key, value in Overall_para.items():
    string_send += f"{key}={value}\r\n"

for i, event_group in EG_para.items():
    string_send += f"[event_o_{i+1}]\r\n"
    for key, value in event_group.items():
        string_send += f"{key}={value}\r\n"
string_recv = socket_send_recv('<SET_PARAMETERS_CONTENT>' + string_send)
#time.sleep(5)
is_warning = socket_wait_for_clear_error('Parameter apply is not completed.')

is_prf_modify = Check_PRF(EG_para, Overall_para['EventGroup_num'])
if (is_prf_modify or is_warning):
    Result = input(">> Would you like to proceed? (y/n): ").strip().lower()
    if Result not in ("y", "yes"):
        print(">> ---------- user interrupt ----------")
        sys.exit()

Overall_para = update_Overall_para(parametersini_path)

if Overall_para['Compounding'] == 1:
    EventGroup_num = 1
else:
    EventGroup_num = Overall_para['EventGroup_num']

# --- read updated parameters, ex. DataSize ---
Overall_para['DataSizeFrame'] = int(socket_send_recv('<GET_PARAMETERS_OVERALL_CONTENT>DataSizeFrame'))
for idx_evt in range(EventGroup_num):
    EG_para[idx_evt]['SampleNum'] = int(socket_send_recv('<GET_PARAMETERS_EVENTGROUP'+str(idx_evt+1)+'_CONTENT>SampleNum'))
    EG_para[idx_evt]['DataSizeRF'] = int(socket_send_recv('<GET_PARAMETERS_EVENTGROUP'+str(idx_evt+1)+'_CONTENT>DataSizeRF'))
    EG_para[idx_evt]['DataSizeBF'] = int(socket_send_recv('<GET_PARAMETERS_EVENTGROUP'+str(idx_evt+1)+'_CONTENT>DataSizeBF'))
    EG_para[idx_evt]['DataSizeImg'] = int(socket_send_recv('<GET_PARAMETERS_EVENTGROUP'+str(idx_evt+1)+'_CONTENT>DataSizeImg'))
    EG_para[idx_evt]['Focus_Num'] = int(socket_send_recv('<GET_PARAMETERS_EVENTGROUP'+str(idx_evt+1)+'_CONTENT>Focus_Num'))

# --- allocate memory for the receive buffer(TotalData) ---
if (Overall_para['EnProcessDataType'] == 0 or Overall_para['EnProcessDataType'] == 1):
    # in asynchronous / synchronous mode, allocate one frame size for the receive buffer
    FrameNum = 1
else:
    # in one-shot mode, allocate 'Overall_para['Frames']' frame size for the receive buffer
    FrameNum = Overall_para['Frames']
    
if (Overall_para['DisableCUDA'] == 0):
    if (Overall_para['DataTypetoPlatform'] == 1):      # grayscale image data, uint8
        TotalData = np.zeros(int(Overall_para['DataSizeFrame']) * FrameNum,dtype=np.uint8)
    else:
        if (EG_para[0]['BeamFormingMethod'] == 0):     # RF data, int16
            TotalData = np.zeros(int(Overall_para['DataSizeFrame']/2) * FrameNum,dtype=np.int16)    
        else:                                          # BF data, I+jQ, I and Q are float32
            TotalData = np.zeros(int(Overall_para['DataSizeFrame']/4/2) * FrameNum , dtype=np.complex64)

#  --- Start scan and check error from Prodigy ---
FrameCount = 0
PSRT_dll.ReceivedStatus(FrameCount)
string_recv = socket_send_recv('<START_SCAN>')
if socket_get_error('<START_SCAN>') == 1:
    sys.exit()
    
print('>> ---------- start scan ----------')
ScanStatus = 1
t_start    = 0

# %% creat ui for stop_scan
if Overall_para['Frames'] == 0:
    ui_thread = threading.Thread(target=start_ui, args=(PSRT_dll, Overall_para['EnProcessDataType'], ))
    ui_thread.start()

####################################################################################
#########################     Asynchronous mode     ################################
####################################################################################
if (Overall_para['EnProcessDataType'] == 0):

    print('   >> Asynchronous mode')

    if (Overall_para['Frames'] > 0):
        while (FrameCount < Overall_para['Frames']):
            if (Overall_para['DisableCUDA'] == 0):
                EG_data = Receive_data(PSRT_dll, Overall_para, EG_para, TotalData, FrameCount, FrameNum, EventGroup_num)
            FrameCount += 1
            PSRT_dll.ReceivedStatus(FrameCount)
            print(f'   >> Frame# {FrameCount} done ({1/(timeit.default_timer()-t_start):.2f} fps)')
            t_start = timeit.default_timer()
            if (Overall_para['DisableCUDA'] == 0):        
                User_Application(Overall_para, EG_para, EG_data)
                           
    else: # Frames = 0, continuous scan
        while (ScanStatus == 1):
            ScanStatus = int(socket_send_recv('<CHECK_SCAN_STATUS>'))
            if (ScanStatus == 1):
                if (Overall_para['DisableCUDA'] == 0):
                    EG_data = Receive_data(PSRT_dll, Overall_para, EG_para, TotalData, FrameCount, FrameNum, EventGroup_num)
                FrameCount += 1
                PSRT_dll.ReceivedStatus(FrameCount)
                print(f'   >> Frame# {FrameCount} done ({1/(timeit.default_timer()-t_start):.2f} fps)')
                t_start = timeit.default_timer()
                if (Overall_para['DisableCUDA'] == 0):
                    User_Application(Overall_para, EG_para, EG_data)
                    
            else:
                print(' >> Prodigy stopped scan')

####################################################################################
########################        Synchronous mode       #############################
####################################################################################    
elif (Overall_para['EnProcessDataType'] == 1):
    
    print('   >> Synchronous mode')
    
    while (ScanStatus == 1):
        if (Overall_para['DisableCUDA'] == 0):
            EG_data = Receive_data(PSRT_dll, Overall_para, EG_para, TotalData, 0, FrameNum, EventGroup_num)
        FrameCount += 1
        PSRT_dll.ReceivedStatus(FrameCount)
        print(f'   >> Frame# {FrameCount} done ({1/(timeit.default_timer()-t_start):.2f} fps)')
        t_start = timeit.default_timer()
        time.sleep(0.05)
        if (Overall_para['DisableCUDA'] == 0):
            User_Application(Overall_para, EG_para, EG_data)
        
        ScanStatus = int(socket_send_recv('<CHECK_SCAN_STATUS>'))
        if (FrameCount == Overall_para['Frames']):
            ScanStatus = 2
            # print("StopScan: " + str(PSRT_dll.StopScan(ScanStatus)))
            PSRT_dll.StopScan(ScanStatus)
        if (ScanStatus == 0):
            ScanStatus = 1
            # print("StartScan: " + str(PSRT_dll.StartScan()))
            PSRT_dll.StartScan()


####################################################################################
########################       One shot mode       #################################
####################################################################################    
else:   #EnProcessDataType == 2

   print('   >> One shot mode')
    
   EG_data_all = []
   while (FrameCount < Overall_para['Frames']):
        if (Overall_para['DisableCUDA'] == 0):
            EG_data = Receive_data(PSRT_dll, Overall_para, EG_para, TotalData, FrameCount, FrameNum, EventGroup_num) 
            if (len(EG_data_all) > 0):
                del EG_data_all[0]
            EG_data_all.append(EG_data)    
            FrameCount += Overall_para['Frames']
            PSRT_dll.ReceivedStatus(FrameCount)
        print(f'   >> Frame# {FrameCount} done ({1/(timeit.default_timer()-t_start):.2f} fps)')
        t_start = timeit.default_timer()
        
        if (Overall_para['EnStopWhenMemFull_Platform'] == 0):
            ScanStatus = int(socket_send_recv('<CHECK_SCAN_STATUS>'));
            if (ScanStatus == 0):
                FrameCount = Overall_para['Frames']
            elif (FrameCount == Overall_para['Frames']):
                FrameCount = 0
        if (Overall_para['DisableCUDA'] == 0):
            for idx_frame in range(Overall_para['Frames']):
                EG_data = EG_data_all[0][idx_frame]
                User_Application(Overall_para, EG_para, EG_data)
            
print('>> --------- scan done ----------')
# Scan and recording finished, release the video writer resources
my_recorder.release()
# print('   (if using cv2 to show images, press any key(instead of mouse clicking the "X" button) on cv2 window to close it -----')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(f' >> Ending ScanStatus = {PSRT_dll.CheckScanStatus()}')
