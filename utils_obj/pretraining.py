import yaml
import labelImg
import argparse
import os

if __name__=='__main__':
    general_conf = r'C:\Users\Giulia Ciaramella\PycharmProjects\E2E\general_conf.yaml'
    with open(general_conf) as file:
        d = yaml.full_load(file)
    file.close()

    assets = d['assets']
    i = False
    while not i:
        value = input("Please choose an asset. You can choose among: \n %r \n " % "   ".join(
            map(str, assets.keys())))
        if value not in list(assets.keys()):
            print('Error!The asset you chose is not in the list.')
        else:
            i = True

    # read the path for the proper yaml file
    yaml_file = assets[value]
    with open(yaml_file) as file:
        current_yaml = yaml.full_load(file)
    file.close()

    class_names = current_yaml['names']

    st = str(class_names)
    print(st)
    s = st.replace("'", '')
    s = s.replace("[", '')
    s = s.replace("]", '')
    f = s.split(',')
    classes = [i if not i.startswith(' ') else i.strip() for i in f]

    print(classes)


