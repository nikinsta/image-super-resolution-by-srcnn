import os

# Fonts

COMPARE_IMAGES_RESULT_LABEL_FONT = ('consolas', 12, '')

# OS

OPENFILE_INITIALDIR_WINDOWS = r'C:\Users\User\PycharmProjects\neural-networks-sandbox\saved_images'
SAVEFILE_INITIALDIR_WINDOWS = r'C:\Users\User\PycharmProjects\neural-networks-sandbox\saved_images'
BACKGROUND_IMAGE_PATH_WINDOWS = '../neural-networks-sandbox/resourses/gradient.png'

OPENFILE_INITIALDIR_LINUX = r'/home/nikitaubuntu1604/PycharmProjects/Coursework3/images'
SAVEFILE_INITIALDIR_LINUX = r'/home/nikitaubuntu1604/PycharmProjects/Coursework3/images'
BACKGROUND_IMAGE_PATH_LINUX = '../Coursework3/resourses/gradient.png'

print('Your OS is', os.name)

#TODO os.name => sys.platform

OS_NAME = os.name

if OS_NAME == 'nt':
    OPENFILE_INITIALDIR = OPENFILE_INITIALDIR_WINDOWS
    SAVEFILE_INITIALDIR = SAVEFILE_INITIALDIR_WINDOWS
    BACKGROUND_IMAGE_PATH = BACKGROUND_IMAGE_PATH_WINDOWS
elif OS_NAME == 'posix':
    OPENFILE_INITIALDIR = OPENFILE_INITIALDIR_LINUX
    SAVEFILE_INITIALDIR = SAVEFILE_INITIALDIR_LINUX
    BACKGROUND_IMAGE_PATH = BACKGROUND_IMAGE_PATH_LINUX
else:
    print('UNKNOWN OS')
    exit()