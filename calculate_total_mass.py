import xml.etree.ElementTree as ET
import re

def parse_mass_value(mass_str):
    """
    解析质量值字符串，支持科学计数法和普通数字
    """
    # 移除引号和空格
    mass_str = mass_str.strip().strip('"').strip("'")
    
    # 检查是否为科学计数法格式
    if 'e' in mass_str.lower():
        return float(mass_str)
    else:
        return float(mass_str)

def calculate_total_mass(urdf_file_path):
    """
    计算URDF文件中所有链接的总质量
    """
    # 解析URDF文件
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    
    total_mass = 0.0
    link_count = 0
    
    # 遍历所有链接
    for link in root.findall('link'):
        link_name = link.get('name')
        
        # 查找 inertial 元素
        inertial = link.find('inertial')
        if inertial is not None:
            # 查找 mass 元素
            mass_element = inertial.find('mass')
            if mass_element is not None:
                mass_value = mass_element.get('value')
                if mass_value is not None:
                    try:
                        mass = parse_mass_value(mass_value)
                        total_mass += mass
                        link_count += 1
                        print(f"链接 '{link_name}' 的质量: {mass} kg")
                    except ValueError as e:
                        print(f"解析链接 '{link_name}' 的质量时出错: {e}")
                else:
                    print(f"链接 '{link_name}' 没有质量值")
            else:
                print(f"链接 '{link_name}' 没有质量元素")
        else:
            print(f"链接 '{link_name}' 没有惯性属性")
    
    return total_mass, link_count

if __name__ == "__main__":
    urdf_file_path = "models/physics.urdf"
    total_mass, link_count = calculate_total_mass(urdf_file_path)
    
    print("\n" + "="*50)
    print(f"总计找到 {link_count} 个链接")
    print(f"角色总质量: {total_mass:.6f} kg")
    print("="*50)