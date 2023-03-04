
import openpyxl

def excel_writer(data, path):
    # data: {sheet_name: tuple of tuple(rows)}
    wb = openpyxl.Workbook()
    for sheet_name in data:
        ws = wb.create_sheet(sheet_name)
        for row in data[sheet_name]:
            ws.append(row)
    del wb["Sheet"]
    wb.save(path)

data = {"sheet1": ((1,2),(3,4))}
out_dir = "test.xlsx"
excel_writer(data, out_dir)