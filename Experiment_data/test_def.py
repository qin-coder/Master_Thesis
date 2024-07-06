import subprocess
import os
import json


def find_specific_class_files(directory, classes):
    specific_class_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.class'):
                class_path = os.path.join(root, file)
                # Construct full class name
                class_name = os.path.splitext(os.path.relpath(class_path, directory).replace(os.sep, '.'))[0]
                # Check if the class is in the list of classes to run
                if class_name in classes:
                    specific_class_files.append(class_path)
    return specific_class_files


def run_evosuite_for_classes(classes, project_path):
    for class_path in classes:

        class_name = os.path.splitext(os.path.relpath(class_path, project_path).replace(os.sep, '.'))[0]


        for i in range(20):
            command = ['java', '-jar', 'evosuite-shaded-1.2.1-SNAPSHOT.jar', '-class', class_name, '-projectCP',
                       project_path,
                       '-Dconfiguration_id=Default', '-Dtimeline_interval=10000','-Doutput_variables',
                       'configuration_id,TARGET_CLASS,criterion,Coverage,Total_Goals,Covered_Goals,Size,Length,Total_Time,CoverageTimeline,FitnessTimeline,Implicit_MethodExceptions']

            subprocess.run(command)



with open('subjects.json', 'r') as json_file:
    subjects_data = json.load(json_file)


for project_name, project_info in subjects_data.items():
    project_path = project_info['path']
    classes_to_run = project_info['classes']


    specific_class_files = find_specific_class_files(project_path, classes_to_run)

    run_evosuite_for_classes(specific_class_files, project_path)
