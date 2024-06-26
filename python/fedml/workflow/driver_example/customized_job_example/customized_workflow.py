import os

import fedml
from fedml.workflow import JobStatus, WorkflowType, Workflow, ModelDeployJob, ModelInferenceJob, TrainJob
from typing import List
import argparse

MY_API_KEY = ""  # Here you need to set your default API key from nexus.fedml.ai
MY_USER_NAME = ""


def show_workflow_metadata(workflow):
    # After the workflow finished, print the graph, nodes and topological order
    print("graph", workflow.metadata.graph)
    print("nodes", workflow.metadata.nodes)
    print("topological_order", workflow.metadata.topological_order)
    print("loop", workflow.loop)


def create_deploy_workflow(job_api_key=None, endpoint_name=None):
    # Define the job yaml
    working_directory = os.path.dirname(os.path.abspath(__file__))
    deploy_image_job_yaml = os.path.join(working_directory, "deploy_image_job.yaml")

    # Load the job yaml and change some config items.
    # deploy_image_job_yaml_obj["computing"]["resource_type"] = "A100-80GB-SXM"
    # deploy_image_job_yaml_obj["computing"]["device_type"] = "GPU"
    # ModelDeployJob.generate_yaml_doc(deploy_image_job_yaml_obj, deploy_image_job_yaml)

    # Generate the job object
    deploy_image_job = ModelDeployJob(
        name="deploy_image_job", endpoint_name=endpoint_name,
        job_yaml_absolute_path=deploy_image_job_yaml, job_api_key=job_api_key)
    deploy_image_job.config_version = fedml.get_env_version()

    # Define the workflow
    workflow = Workflow(
        name="deploy_workflow", loop=False,
        api_key=job_api_key, workflow_type=WorkflowType.WORKFLOW_TYPE_DEPLOY)

    # Add the job object to workflow and set the dependencies (DAG based).
    workflow.add_job(deploy_image_job)

    # Deploy the workflow
    workflow.deploy()

    # Run workflow
    workflow.run()

    # Get the status and result of workflow
    workflow_status = workflow.get_workflow_status()
    workflow_output = workflow.get_workflow_output()
    all_jobs_outputs = workflow.get_all_jobs_outputs()
    print(f"Final status of the workflow is as follows. {workflow_status}")
    print(f"Output of the workflow is as follows. {workflow_output}")
    print(f"Output of all jobs is as follows. {all_jobs_outputs}")

    return workflow_status, workflow_output


def create_inference_workflow(
        job_api_key=None, endpoint_name_list: List[str] = None, input_json=None, user_name=None):
    # Generate the job object
    inference_jobs = list()
    for index, endpoint_name in enumerate(endpoint_name_list):
        inference_job = ModelInferenceJob(
            name=f"inference_job_{index}", endpoint_name=endpoint_name, job_api_key=job_api_key,
            endpoint_user_name=user_name
        )
        inference_jobs.append(inference_job)

    # Define the workflow
    workflow = Workflow(name="inference_workflow", loop=False,
                        api_key=job_api_key, workflow_type=WorkflowType.WORKFLOW_TYPE_DEPLOY)

    # Add the job object to workflow and set the dependency (DAG based).
    for index, inference_job in enumerate(inference_jobs):
        if index == 0:
            workflow.add_job(inference_job)
        else:
            workflow.add_job(inference_job, dependencies=[inference_jobs[index - 1]])

    # Deploy the workflow
    workflow.deploy()

    # Set the input to the workflow
    input_json = {"arr": [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.00005000e-02,-1.00005000e-02,-1.00005000e-02,-1.39737990e-02,-1.89315247e-02,-2.31843010e-02,-3.60728861e-02,-3.92619154e-02,-3.80269994e-02,-3.90143887e-02,-3.46046778e-02,-2.57765396e-02,-2.09733754e-02,-2.17809993e-02,-1.44984527e-02,-1.18807892e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.78081425e-02,-2.32058779e-02,-2.98662898e-02,-4.14395151e-02,-5.86512813e-02,-8.12643979e-02,-1.05997038e-01,-1.21704878e-01,-1.34457288e-01,-1.39756261e-01,-1.41562422e-01,-1.35229133e-01,-1.20246727e-01,-1.04490087e-01,-8.70044931e-02,-7.16699334e-02,-4.85892545e-02,-3.24260775e-02,-2.16926329e-02,-1.00005000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.32956624e-02,-2.25936238e-02,-3.83702224e-02,-5.98206019e-02,-8.42014426e-02,-1.18390816e-01,-1.54266827e-01,-1.88282524e-01,-2.19803054e-01,-2.42936317e-01,-2.55020324e-01,-2.59481423e-01,-2.49404582e-01,-2.26727106e-01,-2.00418885e-01,-1.67161170e-01,-1.34317009e-01,-9.58717755e-02,-7.36565245e-02,-5.03983075e-02,-2.69783475e-02,-1.68919000e-02,-1.00005000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.47795885e-02,-2.51221010e-02,-3.81226487e-02,-7.86317321e-02,-1.19593671e-01,-1.65704529e-01,-2.28814281e-01,-2.88620224e-01,-3.54491034e-01,-4.21140618e-01,-4.80243669e-01,-5.27064646e-01,-5.40807419e-01,-5.21388017e-01,-4.74446021e-01,-4.03948632e-01,-3.36571539e-01,-2.71580657e-01,-2.06667410e-01,-1.54539645e-01,-1.08856709e-01,-6.77589146e-02,-3.40327281e-02,-2.15091205e-02, 0.00000000e+00, 0.00000000e+00,-1.00005000e-02,-1.07381289e-02,-2.60253876e-02,-5.70600482e-02,-9.14378767e-02,-1.43000013e-01,-1.99005834e-01,-2.66034404e-01,-3.53401549e-01,-4.50251488e-01,-5.51598332e-01,-6.47939202e-01,-7.43171364e-01,-8.18162561e-01,-8.51073275e-01,-8.31121680e-01,-7.63764496e-01,-6.59992784e-01,-5.47527626e-01,-4.39376979e-01,-3.35576590e-01,-2.54856553e-01,-1.83933732e-01,-1.26755715e-01,-7.06477667e-02,-3.88818206e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.34176155e-02,-3.90612132e-02,-8.73974922e-02,-1.33107017e-01,-1.94532142e-01,-2.74786330e-01,-3.69886454e-01,-4.82920333e-01,-6.05294063e-01,-7.35621386e-01,-8.69509827e-01,-9.89564738e-01,-1.09132506e+00,-1.13182948e+00,-1.09408349e+00,-9.96373436e-01,-8.68781173e-01,-7.17778845e-01,-5.70649327e-01,-4.39021868e-01,-3.26889344e-01,-2.35934504e-01,-1.67697996e-01,-9.95100269e-02,-4.79392976e-02,-1.87851186e-02, 0.00000000e+00,-1.17322667e-02,-2.88274493e-02,-6.46532861e-02,-1.18956716e-01,-1.77837580e-01, 1.53795878e+00, 2.57176245e+00, 1.53212043e+00, 1.00392168e+00,-1.79355647e-01,-5.91732991e-01,-1.05273662e+00,-1.15378689e+00,-1.22142979e+00,-1.23881560e+00,-1.21321586e+00,-1.14302847e+00,-1.02018313e+00,-8.57098743e-01,-6.76706697e-01,-5.16203262e-01,-3.79287244e-01,-2.71402545e-01,-1.89934521e-01,-1.19940614e-01,-5.56340911e-02,-1.45752163e-02, 0.00000000e+00,-2.06611389e-02,-4.37166621e-02,-8.08756237e-02,-1.40488164e-01,-2.07699245e-01, 3.77477260e+00, 3.14033146e+00, 2.28939169e+00, 1.76127332e+00, 1.43185420e+00, 1.13131350e+00, 6.79164893e-01, 6.65484747e-01, 6.66043389e-01, 6.80680095e-01, 6.77305174e-01, 6.65508286e-01, 7.21340316e-01, 8.83661589e-01, 9.17518690e-01, 2.82541074e-02,-4.01002939e-01,-2.83099723e-01,-1.94831338e-01,-1.23075256e-01,-6.66126860e-02,-1.61462821e-02,-1.12546885e-02,-2.93918605e-02,-4.84646663e-02,-9.31783260e-02,-1.46682925e-01,-2.18121209e-01, 8.30460131e-01, 1.04725853e+00, 1.47086928e-01, 2.59684517e-01, 4.95679969e-01, 9.98953721e-01, 1.29535061e+00, 1.12204782e+00, 1.41528197e+00, 1.42599520e+00, 1.36416372e+00, 1.22805443e+00, 1.03395727e+00, 1.40874227e+00, 1.73166837e+00, 1.00260058e+00,-4.01823716e-01,-2.75049233e-01,-1.81713744e-01,-1.07567122e-01,-5.66041118e-02,-1.89159236e-02,-1.21427928e-02,-2.43168731e-02,-5.02703770e-02,-8.87358114e-02,-1.38806025e-01,-2.12706019e-01,-3.21729999e-01,-4.62313723e-01,-6.52442841e-01,-8.45524923e-01,-9.61258323e-01,-7.93125052e-01,-2.26359955e-01,-6.40468216e-01,-1.23720090e-01,-1.67157468e-01,-2.55843161e-01,-4.41448335e-01,-7.92766628e-01, 1.30597044e+00, 1.81460411e+00, 6.91054579e-01,-3.83665051e-01,-2.63105130e-01,-1.66473946e-01,-7.99663431e-02,-4.55007946e-02,-1.95541446e-02,-1.00005000e-02,-1.86206584e-02,-4.14986832e-02,-7.22615997e-02,-1.23238725e-01,-2.12256343e-01,-3.31309824e-01,-4.91126078e-01,-6.87704902e-01,-8.62602670e-01,-9.39124713e-01,-8.69991467e-01,-7.58168797e-01,-7.22198511e-01,-7.39826964e-01,-8.09980626e-01,-9.11188613e-01,-1.00032001e+00,-2.21550751e-01, 1.53134484e+00, 1.47605194e+00,-2.73150738e-01,-3.63157263e-01,-2.52975575e-01,-1.57152039e-01,-6.52009258e-02,-3.35283586e-02,-1.24209728e-02, 0.00000000e+00,-1.48492790e-02,-3.29699917e-02,-6.01451792e-02,-1.18353377e-01,-2.19271688e-01,-3.54392407e-01,-5.23006773e-01,-7.15682870e-01,-8.62626101e-01,-9.05242890e-01,-8.31592288e-01,-7.51312636e-01,-7.62948163e-01,-8.25877849e-01,-9.30232292e-01,-1.04727288e+00,-8.79016953e-01, 1.11455708e+00, 1.61660969e+00, 2.64000765e-01,-4.64282235e-01,-3.54907482e-01,-2.56014147e-01,-1.58427696e-01,-6.20647188e-02,-2.42921899e-02, 0.00000000e+00, 0.00000000e+00,-1.17874599e-02,-2.52632841e-02,-5.02423656e-02,-1.15068847e-01,-2.35195531e-01,-3.77531303e-01,-5.47311188e-01,-7.23069536e-01,-8.48981953e-01,-8.78897369e-01,-8.26469482e-01,-7.95496372e-01,-8.83536617e-01,-9.94814123e-01,-1.13364619e+00,-1.20871511e+00, 5.60198157e-05, 1.28700658e+00, 1.50082995e+00,-1.22561277e-01,-4.62110102e-01,-3.60151562e-01,-2.63898374e-01,-1.66295096e-01,-5.68635009e-02,-1.05441394e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.66367790e-02,-4.23254862e-02,-1.19931644e-01,-2.52550583e-01,-3.91916340e-01,-5.56171069e-01,-7.17849905e-01,-8.29516019e-01,-8.54549188e-01,-8.45989670e-01,-8.89246054e-01,-1.03761315e+00,-1.16457617e+00,-1.30025654e+00,-7.40699086e-01, 1.05188993e+00, 1.30369880e+00,-1.63440609e-01,-5.90584640e-01,-4.74233049e-01,-3.68789557e-01,-2.74082099e-01,-1.74264813e-01,-6.96188843e-02,-1.80031510e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.68610568e-02,-4.51688568e-02,-1.31668459e-01,-2.67838929e-01,-3.98906806e-01,-5.48202377e-01,-6.90077015e-01,-7.89823563e-01,-8.31599129e-01,-8.61314493e-01,-9.56815660e-01,-1.11036634e+00,-1.22743073e+00,-1.31006468e+00,-2.57368600e-02, 1.14239899e+00, 7.61423491e-01,-7.06825874e-01,-6.08999426e-01,-4.92457882e-01,-3.80502867e-01,-2.79282191e-01,-1.73984018e-01,-7.67235054e-02,-1.95871373e-02,-1.00005000e-02, 0.00000000e+00,-1.00005000e-02,-2.48178080e-02,-5.52275065e-02,-1.48243512e-01,-2.83202341e-01,-4.02212500e-01,-5.34598048e-01,-6.56007943e-01,-7.38083794e-01,-7.81657503e-01,-8.24620535e-01,-9.18824463e-01,-1.04078449e+00,-1.13391454e+00,-1.09212795e+00, 7.05920310e-01, 1.17679031e+00,-3.73781820e-01,-7.58547572e-01,-6.28680640e-01,-5.01492113e-01,-3.81043892e-01,-2.70505206e-01,-1.68251255e-01,-7.84168728e-02,-2.27999680e-02,-1.57856413e-02, 0.00000000e+00, 0.00000000e+00,-2.69850288e-02,-6.76999793e-02,-1.67498207e-01,-2.98089736e-01,-4.11096027e-01,-5.22810883e-01,-6.25838621e-01,-6.93423683e-01,-7.31704263e-01,-7.67086709e-01,-8.29980030e-01,-9.21590434e-01,-1.00562716e+00, 7.79492952e-02, 1.22959017e+00, 6.36500653e-01,-9.01400043e-01,-7.69630793e-01,-6.35363773e-01,-4.94618472e-01,-3.69117095e-01,-2.55794246e-01,-1.56732083e-01,-7.83809414e-02,-2.67109338e-02,-1.48726634e-02, 0.00000000e+00,-1.00005000e-02,-3.48385687e-02,-8.69311199e-02,-1.85622432e-01,-3.11777198e-01,-4.27690033e-01,-5.30457702e-01,-6.12837575e-01,-6.69073252e-01,-7.06628103e-01,-7.37178903e-01,-7.79583917e-01,-8.66698428e-01,-2.88157768e-01, 1.21930590e+00, 1.10500698e+00,-5.04139890e-01,-9.09137779e-01,-7.74520432e-01,-6.19405771e-01,-4.72096102e-01,-3.44822207e-01,-2.35626373e-01,-1.44455008e-01,-7.69092863e-02,-2.86146987e-02,-1.00005000e-02, 0.00000000e+00,-1.00005000e-02,-3.42628198e-02,-1.01174053e-01,-1.95711272e-01,-3.24606261e-01,-4.42716711e-01,-5.45960978e-01,-6.37281741e-01,-7.03742928e-01,-7.53441795e-01,-7.88772419e-01,-8.29773267e-01,-7.45526297e-01, 9.49893727e-01, 1.18293215e+00, 3.85795002e-01,-1.02329900e+00,-8.98728840e-01,-7.36858006e-01,-5.75258663e-01,-4.30322485e-01,-3.09120250e-01,-2.09889823e-01,-1.31895170e-01,-7.31506415e-02,-2.76674735e-02,-1.00005000e-02, 0.00000000e+00,-1.00005000e-02,-4.00234981e-02,-1.07093740e-01,-1.94645695e-01,-3.16981297e-01,-4.40895564e-01,-5.60086039e-01,-6.67605659e-01,-7.63806998e-01,-8.43535003e-01,-9.03604039e-01,-9.38010529e-01, 7.63887624e-01, 1.12176928e+00, 7.84111000e-01,-8.18046093e-01,-9.91046672e-01,-8.28340182e-01,-6.52780006e-01,-4.95325185e-01,-3.64891317e-01,-2.61772085e-01,-1.75298870e-01,-1.12966586e-01,-6.17374486e-02,-2.70715466e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-4.06825662e-02,-9.78606438e-02,-1.77848987e-01,-2.87783481e-01,-4.12614752e-01,-5.43271605e-01,-6.71018812e-01,-7.98159188e-01,-9.16686263e-01,-1.02499517e+00,-7.73682132e-01, 1.09355574e+00, 1.05041156e+00,-4.98209852e-01,-1.05256459e+00,-8.70980804e-01,-6.88431167e-01,-5.23166414e-01,-3.91308572e-01,-2.82035183e-01,-1.99071147e-01,-1.36525170e-01,-8.93688913e-02,-4.13170860e-02,-1.68508310e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-2.83386899e-02,-7.65120563e-02,-1.41969555e-01,-2.32658498e-01,-3.41261378e-01,-4.69723228e-01,-6.06194512e-01,-7.47366354e-01,-8.80786554e-01,-7.29389144e-01, 8.95224865e-01, 1.11943124e+00,-1.05438374e-01,-1.00783177e+00,-8.59696548e-01,-6.83890026e-01,-5.31181637e-01,-3.95889778e-01,-2.89956123e-01,-2.03267966e-01,-1.42951450e-01,-9.63532989e-02,-6.43914026e-02,-3.37070214e-02,-1.11853003e-02, 0.00000000e+00, 0.00000000e+00,-1.00005000e-02,-1.51722732e-02,-4.80051146e-02,-9.51161616e-02,-1.60643556e-01,-2.45453283e-01,-3.53245922e-01,-4.74265429e-01,-5.98667391e-01,-7.29305101e-01, 3.89322873e-01, 1.38694264e+00, 1.37486731e+00,-4.03963644e-01,-7.74445930e-01,-6.38730244e-01,-5.02999283e-01,-3.87339921e-01,-2.79971294e-01,-1.98381814e-01,-1.35822721e-01,-9.65383286e-02,-6.33365644e-02,-4.27549534e-02,-2.57581657e-02,-1.00005000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-2.37543896e-02,-5.22032466e-02,-8.58749627e-02,-1.40703979e-01,-2.08515621e-01,-2.90149335e-01,-3.68567087e-01, 3.34201602e-01, 2.33307288e+00, 2.27286258e+00, 2.23777229e+00, 4.12218057e-02,-4.94890333e-01,-4.22342015e-01,-3.39048837e-01,-2.57069088e-01,-1.85534152e-01,-1.36577185e-01,-8.60242391e-02,-5.78259874e-02,-3.36364160e-02,-1.81122384e-02,-1.00005000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.36274661e-02,-2.85803164e-02,-4.74793553e-02,-7.79785591e-02,-1.18532172e-01,-1.67201555e-01,-2.14787719e-01, 2.22171299e+00, 4.30500754e+00, 4.03125111e+00, 3.36505818e+00, 3.79953648e-01,-2.84269948e-01,-2.47694588e-01,-2.05869945e-01,-1.55925102e-01,-1.16435448e-01,-8.57647974e-02,-5.46508166e-02,-4.01800073e-02,-2.37589970e-02,-1.65780693e-02,-1.00005000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.15748833e-02,-2.84271584e-02,-5.06655656e-02,-7.40332846e-02,-1.00455604e-01,-1.24744578e-01, 4.17363552e+00, 7.81243004e+00, 5.78969790e+00, 3.22149281e-01,-1.81506609e-01,-1.60333393e-01,-1.39182079e-01,-1.18875455e-01,-8.73316648e-02,-7.00227708e-02,-5.40690537e-02,-3.84297037e-02,-2.65616274e-02,-1.61844507e-02,-1.19683967e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.32918601e-02,-1.59980455e-02,-2.07236291e-02,-2.66997366e-02,-2.84703819e-02,-3.43035092e-02,-4.10336906e-02,-4.88886427e-02,-5.48357917e-02,-5.51988782e-02,-4.69971082e-02,-3.88769026e-02,-3.16010302e-02,-2.85226846e-02,-2.17365890e-02,-1.00005000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]}
    workflow.set_workflow_input(input_json)

    # Run workflow
    workflow.run()

    # Get the status and result of workflow
    workflow_status = workflow.get_workflow_status()
    workflow_output = workflow.get_workflow_output()
    all_jobs_outputs = workflow.get_all_jobs_outputs()
    print(f"Final status of the workflow is as follows. {workflow_status}")
    print(f"Output of the workflow is as follows. {workflow_output}")
    print(f"Output of all jobs is as follows. {all_jobs_outputs}")

    return workflow_status, workflow_output


def create_train_workflow(job_api_key=None):
    # Define the job yaml
    working_directory = os.path.dirname(os.path.abspath(__file__))
    train_job_yaml = os.path.join(working_directory, "train_job.yaml")

    train_job = TrainJob(name="train_job", job_yaml_absolute_path=train_job_yaml, job_api_key=job_api_key)
    train_job.config_version = fedml.get_env_version()

    # Define the workflow
    workflow = Workflow(name="train_workflow", loop=False,
                        api_key=job_api_key, workflow_type=WorkflowType.WORKFLOW_TYPE_TRAIN)
    workflow.add_job(train_job)

    # Deploy the workflow
    workflow.deploy()

    # Set the input to the workflow
    input_json = {"data": "test input"}
    workflow.set_workflow_input(input_json)

    # Run workflow
    workflow.run()

    # Get the status and result of workflow
    workflow_status = workflow.get_workflow_status()
    workflow_output = workflow.get_workflow_output()
    all_jobs_outputs = workflow.get_all_jobs_outputs()
    print(f"Final status of the workflow is as follows. {workflow_status}")
    print(f"Output of the workflow is as follows. {workflow_output}")
    print(f"Output of all jobs is as follows. {all_jobs_outputs}")

    return workflow_status, workflow_output


if __name__ == "__main__":
    fedml.set_env_version("test")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--deploy", "-d", nargs="*", help="Create a deploy workflow")
    parser.add_argument("--inference", "-i", nargs="*", help='Create a inference workflow')
    parser.add_argument("--train", "-t", nargs="*", help='Create a train workflow')
    parser.add_argument("--endpoint_name", "-e", type=str, default=None, help='Endpoint name for inference')
    parser.add_argument("--api_key", "-k", type=str, default=MY_API_KEY, help='API Key from the Nexus AI Platform')
    parser.add_argument("--user_name", "-u", type=str, default=MY_USER_NAME, help='User name from the Nexus AI Platform')
    parser.add_argument("--infer_json", "-ij", type=str, default=None, help='Input json data for inference')

    args = parser.parse_args()
    is_deploy = args.deploy
    if args.deploy is None:
        is_deploy = False
    else:
        is_deploy = True
    is_inference = args.inference
    if args.inference is None:
        is_inference = False
    else:
        is_inference = True
    is_train = args.train
    if args.train is None:
        is_train = False
    else:
        is_train = True

    workflow_status, outputs = None, None
    deployed_endpoint_name = args.endpoint_name
    if is_deploy:
        workflow_status, outputs = create_deploy_workflow(job_api_key=args.api_key, endpoint_name=args.endpoint_name)
        deployed_endpoint_name = outputs.get("endpoint_name", None)

    if is_inference and deployed_endpoint_name is not None:
        create_inference_workflow(
            job_api_key=args.api_key, endpoint_name_list=[deployed_endpoint_name],
            input_json=args.infer_json, user_name=args.user_name)

    if is_train:
        create_train_workflow(job_api_key=args.api_key)
        exit(0)
