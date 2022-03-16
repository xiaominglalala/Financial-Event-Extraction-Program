import xlrd
import xlwt
import pandas as pd
# 导入需要读取的第一个Excel表格的路径
data1 = xlrd.open_workbook(r'C:\Users\BenWong\Desktop\金融事件提取\数据集\SmoothNLP金融新闻数据集样本20k.xlsx')
data2 = r'C:\Users\BenWong\Desktop\金融事件提取\数据集\_try.xlsx'
table = data1.sheets()[0]
# 创建一个空列表，存储Excel的数据
tables = []



# 将excel表格内容导入到tables列表中
def import_excel(excel):
    for rown in range(excel.nrows):
        array = {'num': '', 'content': '', 'pub_ts': '', '因果关系句': ''}
        test = table.cell_value(rown, 4)
        if test == '':
            continue
        else:
            array['num'] = table.cell_value(rown,0)
            array['content'] = table.cell_value(rown, 2)
            array['pub_ts'] = table.cell_value(rown,3)
            array['因果关系句'] = test
        tables.append(array)


#导出到新的excel表格
def export_excel(export):
   #将字典列表转换为DataFrame
   pf = pd.DataFrame(list(export))
   #指定字段顺序
   order = ['num','content','pub_ts','因果关系句']
   pf = pf[order]

   #指定生成的Excel表格名称
   file_path = pd.ExcelWriter('new_smooth.xlsx')
   #替换空单元格
   pf.fillna(' ',inplace = True)
   #输出
   pf.to_excel(file_path,encoding = 'utf-8',index = False)
   #保存表格
   file_path.save()



import_excel(table)
length = len(tables)
for i in tables:
    print(i)

print("一共标注了" + str(length) + "条数据")
export_excel(tables)
