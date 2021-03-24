import os, json

class TFconfig():
    def __init__(self, task_id=0 ):
        self.task_id = task_id
        self.tf_config_str = ""

    def generate_config(self, config, ip_list=['localhost'], base_port=3333):
        chiefs=[]
        workers=[]

        for i in range(config.num_node):
                temp_str = '%s:%04d'%(ip_list[int(i)],\
                        (base_port + i ))
                workers.append(temp_str)

        # Cluster
        self.tf_config_str = "{\"cluster\": {"

        ## Chief
        if len(workers) > 1:
            self.tf_config_str += "\"chief\" : ["
            self.tf_config_str += "\"%s\""%(workers[0])
            self.tf_config_str +="], "

        # Worker
        if workers is not None:
            self.tf_config_str += "\"worker\" : ["
            for i, w in enumerate(workers):
                self.tf_config_str += "\"%s\""%(w)
                if i != (len(workers)-1):
                    self.tf_config_str += ", "
            self.tf_config_str +="]"
        self.tf_config_str +="}, "

        # Task
        self.tf_config_str +="\"task\":"
        self.tf_config_str +="{\"type\": \"%s\", \"index\": %d}"%('worker', self.task_id)

        # End
        self.tf_config_str +="}"

        self.tf_config_dict = json.loads(self.tf_config_str)
        return self.tf_config_dict

    def set_rank(self, task_id):
        self.tf_config_dict['task']['index'] = task_id

    def clear_config(self):
        os.environ.pop('TF_CONFIG', None)

    def reset_config(self):
        self.clear_config()
        os.environ['TF_CONFIG'] = self.tf_config_str

    def set_config_custrom_str(self, custom_config_str):
        self.clear_config()
        os.environ['TF_CONFIG'] = custom_config_str

    def set_config_custom_dict(self, custom_config_dict):
        self.clear_config()
        os.environ['TF_CONFIG'] = json.dumps(custom_config_str)

    def print_config(self):
        print("\n\n[%d]=======================      Network Configuration      ======================="%self.task_id)
        print("[%d]%s"%(self.task_id, self.tf_config_str))
        print("[%d]===============================================================================\n\n"%self.task_id)

