from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QFont
from PyQt5.QtWidgets import QItemDelegate, QTableWidgetItem, QAbstractButton
from numpy import int64


class PandasModel(QtCore.QAbstractTableModel):

    def __init__(self, data, parent=None):
        """

        :param data: a pandas dataframe
        :param parent:
        """
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        # self.headerdata = data.columns


    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return '{:.5f}'.format(self._data.values[index.row()][index.column()])
            if role == Qt.BackgroundRole:
                if self._data.values[index.row()][1] > 0:
                    return QBrush(Qt.yellow)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # print(self._data.columns[rowcol])
        # print(self._data.index[rowcol])
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[section]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[section]
        return None

    def flags(self, index):
        flags = super(self.__class__, self).flags(index)
        flags |= QtCore.Qt.ItemIsEditable
        flags |= QtCore.Qt.ItemIsSelectable
        flags |= QtCore.Qt.ItemIsEnabled
        flags |= QtCore.Qt.ItemIsDragEnabled
        flags |= QtCore.Qt.ItemIsDropEnabled
        return flags

    def sort(self, column, order=Qt.AscendingOrder):
        """Sort table by given column number.
        """
        try:
            if column == -1:
                self.layoutAboutToBeChanged.emit()
                self._data = self._data.sort_index()
                self.layoutChanged.emit()
            else:
                self.layoutAboutToBeChanged.emit()
                self._data = self._data.sort_values(self._data.columns[column], ascending=(not order))
                self.layoutChanged.emit()
        except Exception as e:
            print(e)


class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout()
        self.pathLE = QtWidgets.QLineEdit(self)
        hLayout.addWidget(self.pathLE)
        self.loadBtn = QtWidgets.QPushButton("Select File", self)
        hLayout.addWidget(self.loadBtn)
        vLayout.addLayout(hLayout)
        self.pandasTv = QtWidgets.QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.loadBtn.clicked.connect(self.load_file)
        self.pandasTv.setSortingEnabled(True)

        # enable upper left button to reset sorting
        btn = self.pandasTv.findChild(QAbstractButton)
        if btn:
            btn.disconnect()
            btn.clicked.connect(self.disableSorting)
        self.pandasTv.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

        self.last_sorted = None

        self.pandasTv.horizontalHeader().sectionClicked.connect(self.test_method)

    def test_method(self, h):
        order = self.pandasTv.horizontalHeader().sortIndicatorOrder()
        if order == 1:
            self.last_sorted = h
        else:
            if h == self.last_sorted:
                self.disableSorting()
                self.last_sorted = None
            else:
                self.last_sorted = None

    def disableSorting(self):
        self.pandasTv.model().sort(-1)
        self.pandasTv.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.pandasTv.clearSelection()

    def load_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "Pickle Files (*.pk)")
        self.pathLE.setText(fileName)
        df = pd.read_pickle(fileName)
        model = PandasModel(df)
        self.pandasTv.setModel(model)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    df = pd.read_pickle(r"D:\Data\Imu_Calib\Messung_19102018\imu_calib_18-10-19\imu-14-10-04.pk")
    n = len(str(abs(df.shape[0])))
    if df.index.dtype == int64:
        df.index = ['{{:0{:d}d}}'.format(n).format(i) for i in df.index.to_list()]
    df.columns = [c+'_0' for c in df.columns]
    model = PandasModel(df)
    w.pandasTv.setModel(model)
    w.pandasTv.setFont(QFont("DejaVu Sans Mono"))
    w.pandasTv.horizontalHeader().setFont(QFont("DejaVu Sans Mono"))
    # w.pandasTv.verticalHeader().setMinimumWidth(w.pandasTv.columnWidth(1))
    w.show()
    sys.exit(app.exec_())