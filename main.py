import tkinter
from tkinter import *
from tkinter.filedialog import askopenfile, askopenfilename, asksaveasfilename
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageTk

import image_handler
from resourses.constants import *
from skimage.measure import compare_psnr, compare_ssim, compare_mse

import numpy as np

first_image = None
second_image = None

first_canvas = None
second_canvas = None


def do_nothing():
    pass


def clear_canvases():
    global first_canvas
    global second_canvas

    global first_image
    global second_image

    first_image = None
    second_image = None

    first_canvas.delete(ALL)
    second_canvas.delete(ALL)


def compare_images():
    global first_image
    global second_image

    if first_image is None:
        return
    if second_image is None:
        return

    window = Toplevel()
    window.title('Результаты измерений')

    footer = Frame(window)
    footer.pack(side=BOTTOM, expand=NO, fill=X)

    ok_button = ToolbarButton(footer, text='ОК')
    ok_button.pack(side=TOP, fill=Y, padx=10, pady=10)
    ok_button.config(command=window.destroy)
    ok_button.config(width=20)

    true_image = image_handler.convert_to_rgb(first_image)
    test_image = image_handler.convert_to_rgb(second_image)
    true_image_data = image_handler.get_image_data(true_image)
    test_image_data = image_handler.get_image_data(test_image)
    true_image_data = np.array(true_image_data, dtype='uint8')
    test_image_data = np.array(test_image_data, dtype='uint8')

    # print(true_image_data.shape)
    # print(test_image_data.shape)

    psnr = compare_psnr(true_image_data, test_image_data)
    ssim = compare_ssim(true_image_data, test_image_data, multichannel=True)
    mse = compare_mse(true_image_data, test_image_data)
    # print(true_image_data.shape)
    # print(image_handler.get_image_head_data(first_image))

    label_text = 'PSNR (peak signal-to-noise ratio) (Пиковое отношение сигнала к шуму): \n' + str(psnr) + ' dB' + '\n' + \
                 'SSIM (structural similarity) (Индекс структурного сходства) \n' + str(ssim) + '\n' + \
                 'MSE (mean squared error) (Среднеквадратичная ошибка)\n' + str(mse)

    # TODO IFC (Information Fidelity Criterion), NQM (Noise Quality Measure),
    # TODO PSNR (weighted peak signal-to-noise ratio),
    # TODO MSSSIM (multiscale structure similarity index)

    label = Label(window, text=label_text, font=COMPARE_IMAGES_RESULT_LABEL_FONT)
    label.pack(side=TOP, fill=BOTH, expand=YES, padx=20, pady=20)

    window.protocol('WM_DELETE_WINDOW', lambda: None)
    window.grab_set()
    window.focus_set()
    window.wait_window()


def open_image():
    print('open_image_button pressed')

    global first_canvas

    filename = askopenfilename(initialdir=OPENFILE_INITIALDIR,
                               initialfile='image.png',
                               title='Открыть изображение',
                               filetypes=[('all files', '.*'), ('png files', '.png'), ('jpg files', '.jpg')],
                               defaultextension='.*')

    if filename == '':
        return

    global first_image
    first_image = Image.open(filename)
    print('Image', filename.rpartition('/')[2], 'opened')

    # print(image.size)
    image = ImageTk.PhotoImage(first_image)

    # print(image.width(), image.height())

    first_canvas.delete(ALL)
    first_canvas.config(scrollregion=(0, 0, image.width(), image.height()))
    first_canvas.create_image(0, 0, image=image, anchor=NW)
    first_canvas.image = image
    first_canvas.focus_set()

    # image = PhotoImage(file=filename)


def open_left_image():
    print('open_left_image running...')
    open_image()


def open_right_image():
    print('open_right_image running...')

    global second_canvas

    filename = askopenfilename(initialdir=OPENFILE_INITIALDIR,
                               initialfile='image.png',
                               title='Открыть изображение',
                               filetypes=[('png files', '.png'), ('jpg files', '.jpg')],
                               defaultextension='.png')

    if filename == '':
        return

    global second_image
    second_image = Image.open(filename)
    print('Image', filename.rpartition('/')[2], 'opened')

    # print(image.size)
    image = ImageTk.PhotoImage(second_image)

    # print(image.width(), image.height())

    second_canvas.delete(ALL)
    second_canvas.config(scrollregion=(0, 0, image.width(), image.height()))
    second_canvas.create_image(0, 0, image=image, anchor=NW)
    second_canvas.image = image
    second_canvas.focus_set()

    # image = PhotoImage(file=filename)


def save_image():
    print('save_image_button pressed')

    global second_image
    if second_image is None:
        return

    global second_canvas

    filename = asksaveasfilename(initialdir=SAVEFILE_INITIALDIR,
                                 initialfile='handled_image.png',
                                 title='Сохранить изображение',
                                 filetypes=[('png files', '.png'), ('jpg files', '.jpg')],
                                 defaultextension='.png')

    if filename == '':
        return

    print('saving image path:', filename)

    second_image.save(filename)

    print('Image', filename.rpartition('/')[2], 'saved')


def handle_image():
    print('handle_image_button pressed')

    if first_image is None:
        return

    global second_image

    # second_image = None

    second_image = image_handler.handle_image(first_image)
    # second_image.show()

    # print(image.size)
    image = ImageTk.PhotoImage(second_image)
    # print(image.width(), image.height())

    second_canvas.delete(ALL)
    second_canvas.config(scrollregion=(0, 0, image.width(), image.height()))
    second_canvas.create_image(0, 0, image=image, anchor=NW)
    second_canvas.image = image
    second_canvas.focus_set()


def make_menu(root):
    root_menu = Menu(root)
    root.config(menu=root_menu)

    file_menu = Menu(root_menu, tearoff=False)
    # file_menu.add_command(label='Открыть...', command=do_nothing)
    open_menu = Menu(file_menu, tearoff=False)
    open_menu.add_command(label='В левом окне', command=(lambda: open_left_image()))
    open_menu.add_command(label='В правом окне', command=(lambda: open_right_image()))
    file_menu.add_cascade(label='Открыть изображение...', menu=open_menu)
    file_menu.add_separator()
    file_menu.add_command(label='Настройки...', command=do_nothing)
    file_menu.add_separator()
    file_menu.add_command(label='Выход', command=root.quit)
    root_menu.add_cascade(label='Файл', menu=file_menu)

    report_menu = Menu(root_menu, tearoff=False)
    report_menu.add_command(label='Сгенерировать...', command=(lambda: do_nothing()))
    root_menu.add_cascade(label='Отчет', menu=report_menu)

    tools_menu = Menu(root_menu, tearoff=False)
    tools_menu.add_command(label='Сравнить изображения', command=(lambda: compare_images()))
    tools_menu.add_separator()
    tools_menu.add_command(label='Статистика', command=(lambda: do_nothing()))
    root_menu.add_cascade(label='Инструменты', menu=tools_menu)

    help_menu = Menu(root_menu, tearoff=False)
    help_menu.add_command(label='Просмотреть справку')
    help_menu.add_separator()
    help_menu.add_command(label='О программе')
    root_menu.add_cascade(label='Справка', menu=help_menu)


def make_first_canvas(root):
    global first_canvas
    first_canvas = Canvas(root)  # , bg='#FFC')
    first_canvas.config(scrollregion=(0, 0, 800, 500))

    yscrollbar = Scrollbar(first_canvas)
    yscrollbar.config(command=first_canvas.yview)
    first_canvas.config(yscrollcommand=yscrollbar.set)
    yscrollbar.pack(side=RIGHT, fill=Y)

    xscrollbar = Scrollbar(first_canvas, orient=HORIZONTAL)
    xscrollbar.config(command=first_canvas.xview)
    first_canvas.config(xscrollcommand=xscrollbar.set)
    xscrollbar.pack(side=BOTTOM, fill=X)

    first_canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=20, pady=20)


def make_second_canvas(root):
    global second_canvas
    second_canvas = Canvas(root)  # , bg='#FFC')
    second_canvas.config(scrollregion=(0, 0, 800, 500))

    yscrollbar = Scrollbar(second_canvas)
    yscrollbar.config(command=second_canvas.yview)
    second_canvas.config(yscrollcommand=yscrollbar.set)
    yscrollbar.pack(side=RIGHT, fill=Y)

    xscrollbar = Scrollbar(second_canvas, orient=HORIZONTAL)
    xscrollbar.config(command=second_canvas.xview)
    second_canvas.config(xscrollcommand=xscrollbar.set)
    xscrollbar.pack(side=BOTTOM, fill=X)

    second_canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=20, pady=20)


class ToolbarButton(Button):
    def __init__(self, parent, **options):
        Button.__init__(self, parent, **options)
        self.config(font=('consolas', 12, 'bold'))
        # self.config(fg='black', bg='#FFF')  # FFA
        self.config(fg='black', bg='#BBF')
        # self.config(fg='black', bg='#CCC')
        # self.bind('<>', )
        self.config(activebackground='#DDF')
        # self.bind('<Tab>', self.config(bg='yellow'))
        # self.bind('<Escape>', self.config(bg='#BBF'))
        self.config(height=2)


def make_background(root):
    bg_path = BACKGROUND_IMAGE_PATH
    bi = PhotoImage(file=bg_path)
    # blabel = Label(root, bg='#9AF')
    blabel = Label(root, image=bi)  # , text='asd')  # , bg='#9AF')
    # blabel.place(x=-220, y=-50, relwidth=1.5, relheight=1.3)
    blabel.image = bi
    blabel.place(x=0, y=0, relwidth=1, relheight=1)


def make_toolbar(root):
    toolbar = Frame(root)
    toolbar.pack(side=BOTTOM, fill=X)
    toolbar.config(padx=5, pady=5)
    toolbar.config(bg='#CCF')  # FFA

    open_image_button = ToolbarButton(toolbar, text='Открыть изображение')
    open_image_button.pack(side=LEFT, fill=Y)
    open_image_button.config(command=open_image)

    handle_image_button = ToolbarButton(toolbar, text='Запустить нейронную сеть')
    handle_image_button.pack(side=LEFT, fill=Y)
    handle_image_button.config(command=handle_image)

    save_image_button = ToolbarButton(toolbar, text='Сохранить изображение')
    save_image_button.pack(side=LEFT, fill=Y)
    save_image_button.config(command=save_image)

    compare_images_button = ToolbarButton(toolbar, text='Сравнить изображения')
    compare_images_button.pack(side=LEFT, fill=Y)
    compare_images_button.config(command=compare_images)

    clear_button = ToolbarButton(toolbar, text='Очистить окна')
    clear_button.pack(side=LEFT, fill=Y)
    clear_button.config(command=clear_canvases)


root = Tk()
root.title("Neural network image processing")

if OS_NAME == 'nt':
    root.state('zoomed')
elif OS_NAME == 'posix':
    root.state("normal")  # normal, iconic, or withdrawn
    root.wm_attributes('-zoomed', 1)
# root.state("normal")
root.resizable(width=TRUE, height=TRUE)

make_background(root)
make_toolbar(root)
make_first_canvas(root)
make_second_canvas(root)
make_menu(root)

root.mainloop()
