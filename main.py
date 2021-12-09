#-*- coding: utf-8 -*-
#
#          ITM           #
#        Catedra         #
# David Jimenez Murillo  #

from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import GUI
import time

class GUI(QtWidgets.QMainWindow, GUI.Ui_Form):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.browse_button.clicked.connect(self.get_dir)
        self.generate_button.clicked.connect(self.make_dl)
        self.dir = None

    def get_dir(self):
        self.dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Directory')
        self.path_edit.setText(self.dir)
        return

    def make_dl(self):
        with open('template.txt', 'r') as template:
            
            with open('test.py', 'x') as dl:
                for line in template:
                    if "self.path = 'dir_path'" in line:
                        dl.write(line.replace('dir_path', self.dir))#(''.join(l[:-1], self.dir))
                    elif 'self.dims = (dims)' in line:
                        dl.write(line.replace('(dims)', '({}, {})'.format(self.width_spinbox.value(), self.height_spinbox.value()), -1))
                    elif 'self.ext = []' in line:
                        ext = []
                        s = ', '
                        if self.png_cb.isChecked():
                            ext.append("'png'")
                        if self.jpg_cb.isChecked():
                            ext.append("'jpg'")
                        if self.bmp_cb.isChecked():
                            ext.append("'bmp'")
                        dl.write(line.replace('[]', '[{}]'.format(s.join(ext))))
                    elif 'self.transform = transforms' in line:
                        tr = []
                        s = ', '
                        if self.toGS_cb.isChecked():
                            tr.append('transforms.Grayscale()')
                        if self.toTensor_cb.isChecked():
                            tr.append('transforms.ToTensor()')
                        dl.write(line.replace('transforms', 'transforms.Compose([{}])'.format(s.join(tr))))
                    else:
                        dl.write(line)
        return

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = GUI()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
