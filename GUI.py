# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(371, 350)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 354, 313))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.MainVerticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.MainVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.MainVerticalLayout.setObjectName("MainVerticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.title_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.title_label.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        self.horizontalLayout_5.addWidget(self.title_label)
        self.logo_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.logo_label.setText("")
        self.logo_label.setPixmap(QtGui.QPixmap(":/main/img/itm_logo_small.png"))
        self.logo_label.setObjectName("logo_label")
        self.horizontalLayout_5.addWidget(self.logo_label)
        self.MainVerticalLayout.addLayout(self.horizontalLayout_5)
        self.verticalSelectDatabaseLayout = QtWidgets.QVBoxLayout()
        self.verticalSelectDatabaseLayout.setObjectName("verticalSelectDatabaseLayout")
        self.browse_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.browse_label.setObjectName("browse_label")
        self.verticalSelectDatabaseLayout.addWidget(self.browse_label)
        self.horizontalLayout1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout1.setObjectName("horizontalLayout1")
        self.path_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.path_edit.setObjectName("path_edit")
        self.horizontalLayout1.addWidget(self.path_edit)
        self.browse_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.browse_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.browse_button.setObjectName("browse_button")
        self.horizontalLayout1.addWidget(self.browse_button)
        self.verticalSelectDatabaseLayout.addLayout(self.horizontalLayout1)
        self.MainVerticalLayout.addLayout(self.verticalSelectDatabaseLayout)
        self.verticalDimensionsLayout = QtWidgets.QVBoxLayout()
        self.verticalDimensionsLayout.setObjectName("verticalDimensionsLayout")
        self.dimensions_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.dimensions_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.dimensions_label.setObjectName("dimensions_label")
        self.verticalDimensionsLayout.addWidget(self.dimensions_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.width_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.width_label.setAlignment(QtCore.Qt.AlignCenter)
        self.width_label.setObjectName("width_label")
        self.horizontalLayout.addWidget(self.width_label)
        self.width_spinbox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.width_spinbox.setWrapping(True)
        self.width_spinbox.setMinimum(8)
        self.width_spinbox.setMaximum(256)
        self.width_spinbox.setProperty("value", 32)
        self.width_spinbox.setObjectName("width_spinbox")
        self.horizontalLayout.addWidget(self.width_spinbox)
        self.height_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.height_label.setTextFormat(QtCore.Qt.AutoText)
        self.height_label.setAlignment(QtCore.Qt.AlignCenter)
        self.height_label.setObjectName("height_label")
        self.horizontalLayout.addWidget(self.height_label)
        self.height_spinbox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.height_spinbox.setWrapping(True)
        self.height_spinbox.setSuffix("")
        self.height_spinbox.setMinimum(8)
        self.height_spinbox.setMaximum(256)
        self.height_spinbox.setProperty("value", 32)
        self.height_spinbox.setObjectName("height_spinbox")
        self.horizontalLayout.addWidget(self.height_spinbox)
        self.verticalDimensionsLayout.addLayout(self.horizontalLayout)
        self.MainVerticalLayout.addLayout(self.verticalDimensionsLayout)
        self.extensionsLayout_4 = QtWidgets.QVBoxLayout()
        self.extensionsLayout_4.setObjectName("extensionsLayout_4")
        self.format_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.format_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.format_label.setObjectName("format_label")
        self.extensionsLayout_4.addWidget(self.format_label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.png_cb = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.png_cb.setObjectName("png_cb")
        self.horizontalLayout_2.addWidget(self.png_cb)
        self.jpg_cb = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.jpg_cb.setObjectName("jpg_cb")
        self.horizontalLayout_2.addWidget(self.jpg_cb)
        self.bmp_cb = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.bmp_cb.setObjectName("bmp_cb")
        self.horizontalLayout_2.addWidget(self.bmp_cb)
        self.extensionsLayout_4.addLayout(self.horizontalLayout_2)
        self.MainVerticalLayout.addLayout(self.extensionsLayout_4)
        self.tranforms_layout = QtWidgets.QVBoxLayout()
        self.tranforms_layout.setObjectName("tranforms_layout")
        self.transforms_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.transforms_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.transforms_label.setObjectName("transforms_label")
        self.tranforms_layout.addWidget(self.transforms_label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.toTensor_cb = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.toTensor_cb.setObjectName("toTensor_cb")
        self.horizontalLayout_3.addWidget(self.toTensor_cb)
        self.toGS_cb = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.toGS_cb.setObjectName("toGS_cb")
        self.horizontalLayout_3.addWidget(self.toGS_cb)
        self.tranforms_layout.addLayout(self.horizontalLayout_3)
        self.MainVerticalLayout.addLayout(self.tranforms_layout)
        self.generate_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.generate_button.setObjectName("generate_button")
        self.MainVerticalLayout.addWidget(self.generate_button)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.title_label.setText(_translate("Form", "Custom Dataloader Generator"))
        self.browse_label.setText(_translate("Form", "Database folder:"))
        self.browse_button.setText(_translate("Form", "Browse"))
        self.dimensions_label.setText(_translate("Form", "Image dimensions:"))
        self.width_label.setText(_translate("Form", "Width"))
        self.height_label.setText(_translate("Form", "Height"))
        self.format_label.setText(_translate("Form", "File extensions:"))
        self.png_cb.setText(_translate("Form", ".png"))
        self.jpg_cb.setText(_translate("Form", ".jpg"))
        self.bmp_cb.setText(_translate("Form", ".bmp"))
        self.transforms_label.setText(_translate("Form", "Pytorch transformations:"))
        self.toTensor_cb.setText(_translate("Form", "toTensor"))
        self.toGS_cb.setText(_translate("Form", "toGrayscale"))
        self.generate_button.setText(_translate("Form", "Generate"))
import resources_rc
