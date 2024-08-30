import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取名为 'demo_1.jpg' 的图像，以灰度模式加载（0 表示灰度模式）。
img0 = cv2.imread('7777.jpg', 0)
#打印图像 img0 的形状（高度和宽度）。
print('img0 shape', img0.shape)
#读取名为 '0-0.jpg' 的模板图像，也以灰度模式加载。
template = cv2.imread('numbers/0-0.jpg', 0)   # 模板可以和原图上的匹配区域 像素 不完全一致，取最小差即可,但对应位置必须一致
#打印模板图像的形状（高度和宽度）。
print('template shape', template.shape)
#获取模板的宽度 (w) 和高度 (h)，这两者在模板匹配中用作矩形框的大小。
w, h = template.shape[: : -1]


# 六种不同模板匹配方法：
# cv.TM_CCOEFF：相关系数匹配方法。
# cv.TM_CCOEFF_NORMED：归一化相关系数匹配方法。
# cv.TM_CCORR：相关匹配方法。
# cv.TM_CCORR_NORMED：归一化相关匹配方法。
# cv.TM_SQDIFF：平方差匹配方法。
# cv.TM_SQDIFF_NORMED：归一化平方差匹配方法。
#methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF ]

#创建图像 img0 的副本，以便在后续操作中保留原始图像。
img = img0.copy()
# 应用模板匹配，使用指定的模板匹配方法在图像 img 中查找模板 template。返回匹配结果的图像 res。
res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
#打印匹配结果图像 res 的形状。
print('res shape', res.shape)

#查找匹配结果图像 res 中的最小值和最大值及其位置。
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
# 在图像上绘制扩展后的矩形
img_expanded = img.copy()
# 在图像上绘制未扩展前的矩形
cv2.rectangle(img_expanded, top_left, (top_left[0] + w, top_left[1] + h), 150, 2)
plt.imshow(img_expanded, cmap='gray')
plt.show()
# 计算扩展后的左上角和右下角坐标
expansion_factor = 5.5
expanded_top_left_x = int(top_left[0] - w * (expansion_factor - 1))
expanded_top_left = (max(expanded_top_left_x, 0), top_left[1])
expanded_bottom_right = (top_left[0] + w, top_left[1] + h)

# 确保右下角坐标不超出原图像范围
expanded_bottom_right = (min(expanded_bottom_right[0], img0.shape[1]), expanded_bottom_right[1])
# 截取扩展后的区域
cropped_region = img0[expanded_top_left[1]:expanded_bottom_right[1], expanded_top_left[0]:expanded_bottom_right[0]]

# 在图像上绘制扩展后的矩形
img_expanded = img.copy()
cv2.rectangle(img_expanded, expanded_top_left, expanded_bottom_right, 255, 2)

# 在图像上绘制未扩展前的矩形
cv2.rectangle(img_expanded, top_left, (top_left[0] + w, top_left[1] + h), 150, 2)
# 保存截取的区域
#cv2.imwrite(f'cropped_matched_region_cv2.TM_CCORR_NORMED.jpg', cropped_region)

#显示匹配结果图像 res，灰度显示
plt.subplot(131), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#显示带有匹配矩形的图像 img，灰度显示，并设置标题为 'Detected Point'。
plt.subplot(132), plt.imshow(img_expanded, cmap='gray')
plt.title('Expanded and Original Rectangles'), plt.xticks([]), plt.yticks([])
cv2.imwrite('img_expanded.jpg', img_expanded)
plt.subplot(133), plt.imshow(cropped_region, cmap='gray')
plt.title('Cropped Matched Region'), plt.xticks([]), plt.yticks([])
cv2.imwrite('cropped_region.jpg', cropped_region)
plt.suptitle(cv2.TM_CCORR_NORMED)
plt.show()

# 进行阈值处理
thres, dst = cv2.threshold(cropped_region, 200, 255, cv2.THRESH_BINARY)
plt.subplot(121)
plt.imshow(dst, cmap='gray')
cv2.imwrite('dst.jpg', dst)


# 定义结构元素（内核），这里使用4x4的矩形内核
kernel = np.ones((4, 4), np.uint8)
# 对二值图像进行膨胀操作
dilated_image = cv2.dilate(dst, kernel, iterations=2)
plt.subplot(122)
plt.title('Dilated Image')
plt.imshow(dilated_image, cmap='gray')
cv2.imwrite('dilated_image.jpg', dilated_image)
plt.show()

# 将dst转换为uint8类型
dst = dilated_image.astype(np.uint8)
# 寻找阈值处理后图像中的轮廓
contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 将dst转换为BGR格式，以便在彩色图像上绘制轮廓
dst_color = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# 绘制所有找到的轮廓，颜色为红色 (0, 0, 255)，线宽为1
cv2.drawContours(dst_color, contours, -1, (0, 0, 255), 1)
# 在Matplotlib中显示带轮廓的彩色图像
plt.imshow(dst_color)
cv2.imwrite('dst_color.jpg', dst_color)
plt.show()
# 打印轮廓的数量
print(len(contours))
# 打印第一个轮廓的面积
print(cv2.contourArea(contours[0]))

# 计算所有轮廓的面积，并存储在area_list中
area_list = [cv2.contourArea(i) for i in contours]
# 计算轮廓面积的平均值，并将其除以3作为阈值thres
thres = np.average(area_list) /3
# 打印计算得到的阈值thres
print("thres:", thres)

#从轮廓中提取区域的边界框，返回最小和最大的x坐标和y坐标
def getArea(contour):
    xl, yl = [], []
    for c in contour:
        xl.append(c[0][0])  # 提取轮廓点的x坐标
        yl.append(c[0][1])  # 提取轮廓点的y坐标
    return np.min(xl), np.max(xl), np.min(yl), np.max(yl)


size_chart = []
for i in contours:
    # 如果轮廓的面积小于阈值thres，则跳过该轮廓
    if cv2.contourArea(i) < thres:
        continue
    # 获取轮廓的最小和最大坐标，并绘制矩形框在dst_color上
    xs, xm, ys, ym = getArea(i)
    cv2.rectangle(dst_color, (xs, ys), (xm, ym), (255, 0, 0))
    # 将矩形框的坐标信息添加到size_chart中
    size_chart.append([[xs, ys], [xm, ym]])

# 显示包含矩形框的彩色图像
plt.subplot(151)
plt.imshow(dst_color)
# 根据矩形框的左上角x坐标对size_chart进行排序
size_chart = sorted(size_chart, key=lambda x: x[0][0])
# 打印排序后的矩形框坐标信息
print(size_chart)


# 初始化x_min和x_max为dst的高度，并初始化delta_y_max为0
x_min = dst.shape[0]
x_max = delta_y_max = 0
# 遍历size_chart以确定x_min、x_max和delta_y_max
for i in size_chart:
    x_min = min(i[0][1], x_min)  # 更新x_min为最小的y坐标
    x_max = max(i[1][1], x_max)  # 更新x_max为最大的y坐标
    dty = i[1][0] - i[0][0]  # 计算矩形框的宽度
    delta_y_max = max(delta_y_max, dty)  # 更新最大宽度

# 调整size_chart中的矩形框坐标
for i in size_chart:
    i[0][1] = x_min - 1  # 将矩形框的左上角y坐标设为x_min - 1
    i[1][1] = x_max + 1  # 将矩形框的右下角y坐标设为x_max + 1
    dty_ = delta_y_max - (i[1][0] - i[0][0])  # 计算需要扩展的宽度
    i[0][0] -= (dty_ - int(dty_ / 2 - 1))  # 扩展左侧
    i[1][0] += int(dty_ / 2 + 1)  # 扩展右侧

# 将灰度图像转换为BGR格式，以便在其上绘制矩形框
wrapped_print = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)
# 在每个调整后的矩形框上绘制紫色矩形
for item in size_chart:
    cv2.rectangle(wrapped_print, item[0], item[1], (255, 0, 255), 1)
    print(item)  # 打印每个矩形框的坐标信息
# 显示包含调整后的矩形框的图像
plt.subplot(152)
plt.imshow(wrapped_print)

# 定义函数用于检测两个矩形是否有重叠
def is_overlap(rect1, rect2):
    x1_min, y1_min = rect1[0]
    x1_max, y1_max = rect1[1]
    x2_min, y2_min = rect2[0]
    x2_max, y2_max = rect2[1]
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


# 初始化一个新的列表来存储调整后的矩形框
adjusted_size_chart = []

# 遍历原始的size_chart
i = 0
while i < len(size_chart):
    item = size_chart[i]
    # 标记当前矩形是否被合并到其他矩形中，默认为False
    merged = False

    # 如果不是倒数第一个框，且与后一个框重叠
    if i < len(size_chart) - 1 and is_overlap(item, size_chart[i + 1]):
        # 如果是倒数第二个框，并且与倒数第一个框重叠
        if i == len(size_chart) - 2:
            # 将倒数第二个框向左移动一段距离
            new_item = [
                [item[0][0] - 20, item[0][1]],
                [item[1][0] - 20, item[1][1]]
            ]
            adjusted_size_chart.append(new_item)
            # 直接将倒数第一个框加入到调整后的列表中，不合并
            adjusted_size_chart.append(size_chart[i + 1])
            merged = True
            i += 1  # 跳过下一个框，因为它已经被处理过了
        else:
            # 合并当前框和后一个框
            merged_item = [
                [min(item[0][0], size_chart[i + 1][0][0]), min(item[0][1], size_chart[i + 1][0][1])],
                [max(item[1][0], size_chart[i + 1][1][0]), max(item[1][1], size_chart[i + 1][1][1])]
            ]
            adjusted_size_chart.append(merged_item)
            merged = True
            i += 1  # 跳过下一个框，因为它已经被合并到当前框中

    # 如果当前框没有被合并到其他矩形中，则加入调整后的矩形列表
    if not merged:
        adjusted_size_chart.append(item)

    i += 1
# 使用调整后的矩形列表替换原始的size_chart
size_chart = adjusted_size_chart
# 将灰度图像转换为BGR格式，以便在其上绘制矩形框
wrapped_print = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)
# 在每个调整后的矩形框上绘制紫色矩形
for item in size_chart:
    cv2.rectangle(wrapped_print, tuple(item[0]), tuple(item[1]), (255, 0, 255), 1)
    print(item)  # 打印每个矩形框的坐标信息
# 显示包含调整后的矩形框的图像
plt.subplot(153)
plt.imshow(wrapped_print)

# 将灰度图像转换为BGR格式，以便在其上绘制矩形框
wrapped_print = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)

# 只保留第一个和最后一个矩形框
if len(size_chart) > 0:
    # 绘制第一个矩形框
    first_rect = size_chart[0]
    top_left = tuple(first_rect[0])
    bottom_right = tuple(first_rect[1])
    cv2.rectangle(wrapped_print, top_left, bottom_right, (255, 0, 255), 1)

if len(size_chart) > 1:
    # 绘制最后一个矩形框
    last_rect = size_chart[-1]
    top_left = tuple(last_rect[0])
    bottom_right = tuple(last_rect[1])
    cv2.rectangle(wrapped_print, top_left, bottom_right, (255, 0, 255), 1)

# 显示新的图像
plt.subplot(154)
plt.imshow(cv2.cvtColor(wrapped_print, cv2.COLOR_BGR2RGB))  # 转换颜色以适应matplotlib


# 获取原始图像的尺寸
height, width = cropped_region.shape
# 将灰度图像转换为BGR格式，以便在其上绘制矩形框
wrapped_print = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)

# 获取第一个和最后一个矩形框
first_rect = size_chart[0]
last_rect = size_chart[-1]
# 矩形的大小
rect_width = first_rect[1][0] - first_rect[0][0]
rect_height = first_rect[1][1] - first_rect[0][1]
# 计算第一个和最后一个矩形框的中心
x1_center = (first_rect[0][0] + first_rect[1][0]) // 2
y1_center = (first_rect[0][1] + first_rect[1][1]) // 2
x2_center = (last_rect[0][0] + last_rect[1][0]) // 2
y2_center = (last_rect[0][1] + last_rect[1][1]) // 2
# 计算矩形框之间的间隔
num_intermediate_rects = 3
interval_x = (x2_center - x1_center) // (num_intermediate_rects + 1)
interval_y = (y2_center - y1_center) // (num_intermediate_rects + 1)

# 准备一个列表来保存矩形的轮廓
rectangles = []

# 函数：将矩形的轮廓添加到列表
def add_rectangle_to_list(rect):
    rectangles.append([
        tuple(rect[0]),  # 左上角
        tuple(rect[1])   # 右下角
    ])

# 绘制第一个矩形
add_rectangle_to_list(first_rect)
cv2.rectangle(wrapped_print, tuple(first_rect[0]), tuple(first_rect[1]), (255, 0, 255), 1)

# 绘制中间的矩形
for i in range(1, num_intermediate_rects + 1):
    x = x1_center + i * interval_x
    y = y1_center + i * interval_y
    new_rect = [
        [x - rect_width // 2, y - rect_height // 2],
        [x + rect_width // 2, y + rect_height // 2]
    ]
    add_rectangle_to_list(new_rect)
    cv2.rectangle(wrapped_print, tuple(new_rect[0]), tuple(new_rect[1]), (255, 0, 255), 1)

# 绘制最后一个矩形
add_rectangle_to_list(last_rect)
cv2.rectangle(wrapped_print, tuple(last_rect[0]), tuple(last_rect[1]), (255, 0, 255), 1)

# 显示新的图像
plt.subplot(155)
plt.imshow(cv2.cvtColor(wrapped_print, cv2.COLOR_BGR2RGB))  # 转换颜色以适应matplotlib
plt.show()

# 准备裁剪图像的列表
cropped_images = []

# 函数：裁剪并调整尺寸
def crop_and_resize(x1, y1, x2, y2, target_size=(200, 280)):
    # 确保裁剪区域在图像范围内
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    # 裁剪图像
    cropped_image = cropped_region[y1:y2, x1:x2]
    # 调整裁剪后的图像尺寸
    resized_image = cv2.resize(cropped_image, target_size)  # (宽度, 高度)
    return resized_image
# 按顺序裁剪矩形区域
# 裁剪第一个矩形
x1, y1 = first_rect[0]
x2, y2 = first_rect[1]
cropped_images.append(crop_and_resize(x1, y1, x2, y2))

# 裁剪中间的矩形
for i in range(1, num_intermediate_rects + 1):
    x = x1_center + i * interval_x
    y = y1_center + i * interval_y
    new_rect = [
        [x - rect_width // 2, y - rect_height // 2],
        [x + rect_width // 2, y + rect_height // 2]
    ]
    x1, y1 = new_rect[0]
    x2, y2 = new_rect[1]
    cropped_images.append(crop_and_resize(x1, y1, x2, y2))

# 裁剪最后一个矩形
x1, y1 = last_rect[0]
x2, y2 = last_rect[1]
cropped_images.append(crop_and_resize(x1, y1, x2, y2))

# 显示绘制矩形框后的图像
fig, axs = plt.subplots(1, len(cropped_images) + 1, figsize=(20, 5))

# 显示绘制了矩形框的图像
axs[0].imshow(cv2.cvtColor(wrapped_print, cv2.COLOR_BGR2RGB))  # 转换BGR图像为RGB
axs[0].axis('off')  # 不显示坐标轴
axs[0].set_title('Image with Rectangles')

# 显示裁剪并调整尺寸后的图像
for i, img in enumerate(cropped_images):
    axs[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换灰度图像为RGB
    axs[i + 1].axis('off')  # 不显示坐标轴
    axs[i + 1].set_title(f'Rectangle {i+1}')

plt.show()
# 裁剪调整后的矩形区域，并调整尺寸为 (200, 280)
x = 0
sub_data_list = []
num_charts = len(rectangles)
plt.figure(figsize=(20, 4))  # 调整图像大小以适应所有子图
for i in rectangles:
    x += 1
    sub_data = dst[i[0][1]: i[1][1], i[0][0]: i[1][0]]
    print("sub_data shape:", sub_data.shape)
    print("sub_data content:", sub_data)
    sub_data = cv2.resize(sub_data, (100, 150))

    sub_data_list.append(sub_data)
    plt.subplot(1, num_charts, x)
    plt.axis('off')
    for j in range(0, 100, 8):  # Lines every 10 columns
        plt.plot([j, j], [0, 150], color='blue')
    for j in range(0, 150, 8):  # Lines every 10 rows
        plt.plot([0, 100], [j, j], color='blue')
    plt.title('Dilated Image')
    plt.imshow(sub_data, cmap='gray')
plt.show()


def seven_blocked_test(img):
    # 初始二进制结果
    res = 0b0000_0000
    # 顶部区域
    top_area_sum = img[0:24, 32:72].sum() / (255 * 24 * 40); res = (res | (top_area_sum > 0.25)) << 1
    # 中部区域
    middle_area_sum = img[64:88, 32:70].sum() / (255 * 24 * 38); res = (res | (middle_area_sum > 0.5)) << 1
    # 底部区域
    bottom_area_sum = img[128:150, 8:63].sum() / (255 * 22 * 55); res = (res | (bottom_area_sum > 0.25)) << 1
    # 左上角区域
    top_left_area_sum = img[24:64, 16:40].sum() / (255 * 40 * 24); res = (res | (top_left_area_sum > 0.2)) << 1
    # 左下角区域
    bottom_left_area_sum = img[88:126, 8:32].sum() / (255 * 38 * 24); res = (res | (bottom_left_area_sum > 0.4)) << 1
    # 右上角区域
    top_right_area_sum = img[27:63, 56:94].sum() / (255 * 36 * 38); res = (res | (top_right_area_sum > 0.3)) << 1
    # 右下角区域
    bottom_right_area_sum = img[87:126, 55:94].sum() / (255 * 39 * 39); res = res | (bottom_right_area_sum > 0.3)

    #小数点区域
    decimal_area_sum = img[122:142, 87:100].sum() / (255 * 20 * 13);has_decimal_point = decimal_area_sum > 0.5
    # 如果小数点区域存在，设置标志位
    if has_decimal_point:
        res |= 0b10000000  # 设置第8位为1
    # 打印二进制结果
    #print(bin(res))
    return res

def special_one_test(img):
    if img[250: 270, 180: 200].sum()/(255*20*20) > 0.30:
        return 1
    else: return "?"

def get_result_from_code(res):
    # 0: 101_1111, 95
    # 1: 000_0011, 3
    # 2: 111_0110, 118
    # 3: 111_0011, 115
    # 4: 010_1011, 43
    # 5: 111_1001, 121
    # 6: 111_1101, 125
    # 6': 011_1101, 61
    # 7: 100_0011, 67
    # 8: 111_1111, 127
    # 9: 111_1011, 123
    # 9', 110_1011, 107

    code_ = {95: 0, 3:1, 118:2, 115:3, 43:4, 121:5, 125:6, 61:6, 67:7, 127:8, 123:9, 107:9,104:'℃'}
    # 获取小数点标志
    has_decimal_point = (res & 0b10000000) != 0
    # 获取数字编码
    try:
        result = code_[res & 0b01111111]  # 忽略第8位的标志
    except KeyError:
        result = special_one_test()

    # 确保result是字符串类型
    result = str(result)

    # 检查是否有小数点
    if has_decimal_point:
        result += "."
    return result

# Initialize results list
results = []
x = 1
for img in sub_data_list:
    # GuassianBlur img
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # set threshold value 40
    _, img = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    # plot the threshold image
    plt.subplot(1, len(sub_data_list), x)
    x += 1
    # plt.axis('off')s
    plt.imshow(img)
    code = seven_blocked_test(img)
    result = get_result_from_code(code)
    results.append(result)
    plt.title(get_result_from_code(code))
plt.show()

# 连接结果为字符串
result_string = ''.join(results)


def validate_and_correct_format(result_string):


    # 检查是否符合正确格式 (两位整数和两位小数)
    pattern_correct = r"^\d{2}\.\d{2}℃$"  # 正确格式
    if re.match(pattern_correct, result_string):
        print(f"识别结果：{result_string}")
        result = result_string[:-1]  # 移除最后的℃
        return result

    # 去掉最后的 ℃ 符号以便处理
    result_string = result_string.strip('℃')
    # 处理包含多个小数点的情况
    # 先去除所有的小数点，然后将剩下的数字进行格式化
    digits_only = re.sub(r'\.', '', result_string)
    if len(digits_only) >= 4:
        corrected_string = f"{digits_only[:2]}.{digits_only[2:4]}℃"
        print(f"修正后的格式: {corrected_string}")
        result = corrected_string[:-1]  # 移除最后的℃
        return result

    # 处理四位数字
    pattern_four_digits = r"^(\d{4})$"
    match_four_digits = re.match(pattern_four_digits, digits_only)
    if match_four_digits:
        number_part = match_four_digits.group(1)
        if len(number_part) == 4:
            corrected_string = f"{number_part[:2]}.{number_part[2:]}℃"
            print(f"修正后的格式: {corrected_string}")
            result = corrected_string[:-1]  # 移除最后的℃
            return result

    # 默认识别错误
    print("识别错误")
    return "0"



saved_result = validate_and_correct_format(result_string)
print(saved_result)

