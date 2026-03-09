import h5py,sys,math,random,os,argparse,tyro,glob,shutil,time,os
import numpy as np
from PIL import Image
from utils import *

import lerobot;print('lerobot ver:', lerobot.__version__, lerobot.__file__)
if lerobot.__version__ == "0.1.0":
    # pip install git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
    
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
else:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset



parser = argparse.ArgumentParser()


parser.add_argument(
    "--repo",
    type=str,
    required=True
)

parser.add_argument(
    "--push",
    action="store_true" 
)

args = parser.parse_args() 
# robomimic_ph__lift_can_square_toolhang_lerobot_ver042_test
REPO_NAME = f"yananchen/{args.repo}"

output_path = HF_LEROBOT_HOME / REPO_NAME # the local path on workstation
if output_path.exists():
    shutil.rmtree(output_path)

print('output_path:', output_path)


if 'mimicgen' in args.repo:
    image_shape = (84, 84, 3) 
    action_dim = 7
elif 'robocasa' in args.repo:
    image_shape = (128, 128, 3)
    action_dim = 12
else:
    image_shape = (256, 256, 3)
    action_dim = 7


dataset = LeRobotDataset.create(
    repo_id=REPO_NAME,
    robot_type="panda",
    fps=10,
    features={
        "image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    },
    image_writer_threads=10,
    image_writer_processes=5,
)






if  'robosuite' in args.repo:
    # robosuite
    files  = glob.glob("/home/yanan/robotics/robosuite/trajectories/*.npz")
    for file in files:
        traj = np.load(file, allow_pickle=True)
        print(file, '===>', len(traj["data"].tolist()))
        # dict_keys(['frontview_image', 'agentview_image', 'wrist_image', 'state', 'actions', 'task'])
        for step in traj["data"].tolist():
            assert 'frontview_image' in step.keys() and 'agentview_image' in step.keys() and 'wrist_image' in step.keys()
            step['image'] = step.pop('agentview_image')
            del step['frontview_image']
            dataset.add_frame(step)
        
        dataset.save_episode()
        print('STATS---> num_episodes:', dataset.num_episodes, 'total frames:', len(dataset))


    # robomimic dataset to lerbot
    '''
    demo_0 ===>
    traj.keys==> actions (59, 7)
    traj.keys==> dones (59,)
    traj.keys==> obs (13,)
            agentview_image (59, 256, 256, 3)
            frontview_image (59, 256, 256, 3)
            object (59, 10)
            robot0_eef_pos (59, 3)
            robot0_eef_quat (59, 4)
            robot0_eef_quat_site (59, 4)
            robot0_eye_in_hand_image (59, 256, 256, 3)
            robot0_gripper_qpos (59, 2)
            robot0_gripper_qvel (59, 2)
            robot0_joint_pos (59, 7)
            robot0_joint_pos_cos (59, 7)
            robot0_joint_pos_sin (59, 7)
            robot0_joint_vel (59, 7)
    traj.keys==> rewards (59,)
    traj.keys==> states (59, 32)

    '''

elif 'robocasa' in args.repo:

    pnp_to_pickplace = {
        "PnPCounterToSink": "PickPlaceCounterToSink",
        "PnPCabToCounter": "PickPlaceCabinetToCounter",
        "PnPStoveToCounter": "PickPlaceStoveToCounter",
        "PnPCounterToStove": "PickPlaceCounterToStove",
        "PnPCounterToCab": "PickPlaceCounterToCabinet",
        "PnPCounterToMicrowave": "PickPlaceCounterToMicrowave",
        "PnPMicrowaveToCounter": "PickPlaceMicrowaveToCounter",
        "PnPSinkToCounter": "PickPlaceSinkToCounter",
    }

    task_to_description_atomic = {
        "PickPlaceCounterToCabinet": "Pick an object from the counter and place it inside the cabinet. The cabinet is already open.",
        "PickPlaceCabinetToCounter": "Pick an object from the cabinet and place it on the counter. The cabinet is already open.",
        "PickPlaceCounterToSink": "Pick an object from the counter and place it in the sink.",
        "PickPlaceSinkToCounter": "Pick an object from the sink and place it on the counter area next to the sink.",
        "PickPlaceCounterToMicrowave": "Pick an object from the counter and place it inside the microwave. The microwave door is already open.",
        "PickPlaceMicrowaveToCounter": "Pick an object from inside the microwave and place it on the counter. The microwave door is already open.",
        "PickPlaceCounterToStove": "Pick an object from the counter and place it in a pan or pot on the stove.",
        "PickPlaceStoveToCounter": "Pick an object from the stove (via a pot or pan) and place it on (the plate on) the counter.",

        "OpenSingleDoor": "Open a microwave door or a cabinet with a single door.",
        "CloseSingleDoor": "Close a microwave door or a cabinet with a single door.",
        "OpenDoubleDoor": "Open a cabinet with two opposite-facing doors.",
        "CloseDoubleDoor": "Close a cabinet with two opposite-facing doors.",

        "OpenDrawer": "Open a drawer.",
        "CloseDrawer": "Close a drawer.",

        "TurnOnSinkFaucet": "Turn on the sink faucet to begin the flow of water.",
        "TurnOffSinkFaucet": "Turn off the sink faucet to begin the flow of water.",
        "TurnSinkSpout": "Turn the sink spout.",

        "TurnOnStove": "Turn on a specified stove burner by twisting the respective stove knob.",
        "TurnOffStove": "Turn off a specified stove burner by twisting the respective stove knob.",

        "CoffeeSetupMug": "Pick the mug from the counter and insert it onto the coffee machine mug holder area.",
        "CoffeeServeMug": "Remove the mug from the coffee machine mug holder and place it on the counter.",

        "CoffeePressButton": "Press the button on the coffee machine to pour coffee into the mug.",
        "TurnOnMicrowave": "Turn on the microwave by pressing the start button.",
        "TurnOffMicrowave": "Turn off the microwave by pressing the stop button.",

        "NavigateKitchen": "Navigate to a specified appliance in the kitchen."
    }

    task_to_description_composite = {
        "PastryDisplay": "Place the pastries on the plates.",
        "OrganizeBakingIngredients": "Place the eggs and milk next to the bowl.",
        "CupcakeCleanup": "Move the fresh-baked cupcake off the tray onto the counter, and place the bowl used for mixing into the sink.",

        "FillKettle": "Open the cabinet, pick the kettle from the cabinet, and place it in the sink.",
        "HeatMultipleWater": "Pick the kettle from the cab and place it on a stove burner. Then pick the pot from the counter and place on another stove burner. Finally, turn both burners on.",
        "VeggieBoil": "Pick up the pot and place it in the sink. Then turn on the sink faucet and let the pot fill up with water. Then turn the sink faucet off and move the pot to the stove. Lastly, turn on the stove and place the food in the pot for boiling.",

        "KettleBoiling": "Pick the kettle from the counter and place it on a stove burner. Then turn the burner on.",
        "ArrangeTea": "Pick the kettle from the counter and place it on the tray. Then pick the mug from the cabinet and place it on the tray. Then close the cabinet doors.",
        "PrepareCoffee": "Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.",

        "ArrangeVegetables": "Pick the vegetables from the sink and place them on the cutting board.",
        "OrganizeVegetables": "Place the vegetables on separate cutting boards.",
        "BreadSetupSlicing": "Place all breads on the cutting board.",
        "ClearingTheCuttingBoard": "Clear the non-vegetable object off the cutting board and place the vegetables onto it.",
        "MeatTransfer": "Retrieve the container from the cabinet, then place the raw meat into the container to avoid contamination.",

        "CondimentCollection": "Pick the condiments from the counter and place them in the cabinet.",
        "DessertAssembly": "Pick up the container with dessert and place it on the tray. Pick up the cupcake and place it on the tray.",
        "ClearingCleaningReceptacles": "Pick the receptacles and place them in the sink. Then turn on the water.",
        "CandleCleanup": "Pick the decorations from the dining table and place them in the open cabinet.",
        "FoodCleanup": "Pick the food items from the counter and place them in the cabinet. Then close the cabinet.",
        "DrinkwareConsolidation": "Pick the drinks from the island and place them in the open cabinet.",
        "BowlAndCup": "Place the cup inside the bowl on the island and move the bowl to any counter.",

        "ThawInSink": "Pick the object from the counter and place it in the sink. Then turn on the sink faucet.",
        "MicrowaveThawing": "Pick the food from the counter and place it in the microwave. Then turn on the microwave.",
        "QuickThaw": "Frozen meat rests on a plate on the counter. Retrieve the meat and place it in a pot on a burner. Then turn the burner on.",
        "DefrostByCategory": "There is a mixed pile of frozen fruits and vegetables on the counter. Locate all the frozen vegetables and place the items in a bowl on the counter. Take all the frozen fruits and defrost them in a running sink.",

        "SetupFrying": "Pick the pan from the cabinet and place it on the stove. Then turn on the stove burner for the pan.",
        "FryingPanAdjustment": "Pick and place the pan from the current burner to another burner and turn the burner on.",
        "MealPrepStaging": "Place both pans onto different burners. Then place the vegetable and the meat on different pans.",
        "AssembleCookingArray": "Move the meat onto the pan on the stove. Then move the condiment and vegetable from the cabinet to the counter where the plate is.",
        "SearingMeat": "Grab the pan from the cabinet and place it on the burner on the stove. Then place the meat on the stove and turn the burner on.",

        "PrepareToast": "Pick the bread, place it on the cutting board, pick the jam, place it on the counter, and close the cabinet.",
        "SweetSavoryToastSetup": "Pick the avocado and bread from the counter and place them on the plate. Then pick the jam from the cabinet and place it next to the plate. Lastly, close the cabinet door.",
        "CheesyBread": "Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish.",
        "BreadSelection": "From the different types of pastries on the counter, select a croissant and place it on the cutting board. Then retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board.",

        "PrepMarinatingMeat": "Pick the meat from its container and place it on the cutting board. Then pick the condiment from the cabinet and place it next to the cutting board.",
        "PrepForTenderizing": "Retrieve a rolling pin from the cabinet and place it next to the meat on the cutting board to prepare for tenderizing.",

        "SetupJuicing": "Open the cabinet, pick all fruits from the cabinet and place them on the counter.",
        "ColorfulSalsa": "Place the avocado, onion, tomato and bell pepper on the cutting board.",
        "SpicyMarinade": "Open the cabinet. Place the bowl and condiment on the counter. Then place the lime and garlic on the cutting board.",

        "HeatMug": "Pick the mug from the cabinet and place it inside the microwave. Then close the microwave.",
        "SimmeringSauce": "Place the pan on the burner on the stove. Then place the tomato and the onion in the pan and turn on the burner.",
        "WaffleReheat": "Open the microwave, place the bowl with waffle inside the microwave, then close the microwave door and turn it on.",
        "WarmCroissant": "Pick the croissant and place it on the pan. Then turn on the stove to warm the croissant.",
        "MakeLoadedPotato": "Retrieve the reheated potato from the microwave, then place it on the cutting board along with cheese and a bottle of condiment.",

        "RestockPantry": "Pick the cans from the counter and place them in their designated side in the cabinet.",
        "StockingBreakfastFoods": "Pick the packaged foods from the counter and place them in the cabinets closest to them.",
        "RestockBowls": "Open the cabinet. Pick the bowls from the counter and place them in the cabinet directly in front. Then close the cabinet.",
        "BeverageSorting": "Sort all alcoholic drinks to one cabinet, and non-alcoholic drinks to the other.",

        "PushUtensilsToSink": "Push the utensils into the sink.",
        "PrepForSanitizing": "Pick the cleaning supplies from the cabinet and place them on the counter.",
        "CleanMicrowave": "Open the microwave. Then pick the sponge from the counter and place it in the microwave.",
        "CountertopCleanup": "Pick the fruit and vegetable from the counter and place them in the cabinet. Then open the drawer and pick the cleaner and sponge from the drawer and place them on the counter.",

        "WineServingPrep": "Open the cabinet directly in front. Then move the alcohol and the cup to the counter with the decoration on it.",
        "ServeSteak": "Pick up the pan with the steak in it and place it on the dining table. Then place the steak on the plate.",
        "PlaceFoodInBowls": "Pick both bowls and place them on the counter. Then pick the food and place it in one bowl and pick the other food and place it in the other bowl.",
        "DessertUpgrade": "Move the dessert items from the plate to the tray.",
        "PrepareSoupServing": "Open the cabinet and move the ladle to the pot. Then close the cabinet.",
        "PanTransfer": "Pick up the pan and dump the vegetables in it onto the plate. Then return the pan to the stove.",

        "DateNight": "Pick up the decoration and the alcohol in the cabinet and move them to the dining counter.",
        "ArrangeBreadBasket": "Open the cabinet, pick up the bread from the cabinet and place it in the bowl. Then move the bowl to the dining counter.",
        "SeasoningSpiceSetup": "Move the condiments from the cabinet to the dining counter.",
        "SetBowlsForSoup": "Move the bowls from the cabinet to the plates on the dining table.",
        "BeverageOrganization": "Move the drinks to the dining counter.",
        "SizeSorting": "Stack the cups/bowls from largest to smallest.",

        "CerealAndBowl": "Open the cabinet. Pick the cereal and bowl from the cabinet and place them on the counter. Then close the cabinet.",
        "BreadAndCheese": "Place the bread and cheese on the cutting board.",
        "YogurtDelightPrep": "Place the yogurt and fruit onto the counter.",
        "MakeFruitBowl": "Open the cabinet. Pick the fruits from the cabinet and place them into the bowl. Then close the cabinet.",
        "VeggieDipPrep": "Place the two vegetables and a bowl onto the tray for setting up a vegetable dip station.",

        "SteamInMicrowave": "Pick the vegetable from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. Then close the microwave door and press the start button.",
        "MultistepSteaming": "Turn on the sink faucet. Then move the vegetable from the counter to the sink. Turn of the sink. Move the vegetable from the sink to the pot next to the stove. Finally, move the pot to the appropriate burner.",
        "SteamVegetables": "Place vegetables into the pot based on the amount of time it would take to steam each, e.g. potatoes and carrots would take the longest. Then turn off the burner beneath the pot.",

        "OrganizeCleaningSupplies": "Open the cabinet. Pick the cleaner and place it next to the sink. Then close the cabinet.",
        "DrawerUtensilSort": "Open the left drawer and push the utensils inside it.",
        "PantryMishap": "Place the vegetable on the counter and the canned food in the drawer. Close the cabinet.",
        "ShakerShuffle": "Pick and place the shaker into the drawer. Then close the cabinet.",
        "SnackSorting": "Place the bar in the bowl and close the drawer.",

        "StackBowlsInSink": "Stack the bowls in the sink.",
        "PreSoakPan": "Pick the pan and sponge and place them into the sink. Then turn on the water.",
        "SortingCleanup": "Pick the mug and place it in the sink. Pick the bowl and place it in the cabinet and then close the cabinet.",
        "DryDrinkware": "A wet mug is on the counter and needs to be dried. Pick it up and place it upside down in the open cabinet.",
        "DryDishes": "Pick the cup and bowl from the sink and place them on the counter for drying.",

        "PrewashFoodAssembly": "Pick the fruit/vegetable from the cabinet and place it in the bowl. Then pick the bowl and place it in the sink. Then turn on the sink facuet.",
        "ClearClutter": "Pick up the fruits and vegetables and place them in the sink. Turn on the sink faucet to wash them. Then turn the sink off and put them in the tray.",
        "DrainVeggies": "Dump the vegetable from the pot into the sink. Then turn on the water and wash the vegetable. Then turn off the water and put the vegetable back in the pot.",
        "AfterwashSorting": "Pick the foods of the same kind from the sink and place them in one bowl. Place the other food in the other bowl. Then turn off the sink faucet."
    }
    task_to_description = task_to_description_atomic | task_to_description_composite

    hdf5_files_single_stage = glob.glob("/home/yanan/robotics/robocasa/datasets/v0.1/single_stage/*/*/*/demo_gentex_im128_randcams.hdf5")
    hdf5_files_multi_stage = glob.glob("/home/yanan/robotics/robocasa/datasets/v0.1/multi_stage/*/*/*/demo_im128.hdf5")
    hdf5_files = hdf5_files_multi_stage + hdf5_files_single_stage
    random.shuffle(hdf5_files)

    for file in hdf5_files:
        assert os.path.exists(file), f"File not found: {file}"
        with h5py.File(file, "r") as f:    
            # data:
            #    demo_0
            #    demo_1
            #    demo_2
            #    ....     
            print('hdf5 file===>', file )   
            print(f['data'].keys())
            print('total trajectories:', len(f['data']))
            assert len(f['data']) >= 50
            for traj_id, traj in f['data'].items():

                print(traj_id, '===>', 'total steps:', traj['actions'].shape[0])
                # print('traj obs==>', traj['obs'].keys())
                # print('image shape:', traj['obs']['robot0_agentview_left_image'].shape,
                #         traj['obs']['robot0_agentview_right_image'].shape,
                #         traj['obs']['robot0_eye_in_hand_image'].shape )


                #['object', 'robot0_agentview_left_image', 'robot0_agentview_right_image', 
                # 'robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 
                # 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 
                #'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']

                assert traj['actions'].shape[0] == traj['states'].shape[0] == traj['rewards'].shape[0] == traj['obs']['robot0_eye_in_hand_image'].shape[0]
                for k, v in traj['obs'].items():
                    assert v.shape[0] == traj['actions'].shape[0]

                for step_ix in range(traj['actions'].shape[0]):
                    
                    # frontview_image = traj['obs']['frontview_image'][()][step_ix]
                    agentview_image = traj['obs']['robot0_agentview_right_image'][()][step_ix]
                    robot0_eye_in_hand_image = traj['obs']['robot0_eye_in_hand_image'][()][step_ix]
                    
                    robot0_eef_pos = traj['obs']['robot0_eef_pos'][step_ix]
                    robot0_eef_quat = traj['obs']['robot0_eef_quat'][step_ix]
                    robot0_gripper_qpos = traj['obs']['robot0_gripper_qpos'][step_ix]
                    
                    ee_state = compute_ee_state(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos) 

                    action = traj['actions'][()][step_ix]
                    
                    assert \
                        isinstance(agentview_image, np.ndarray) and \
                        isinstance(robot0_eye_in_hand_image, np.ndarray) and \
                        isinstance(robot0_eef_pos, np.ndarray)    and \
                        isinstance(robot0_eef_quat, np.ndarray)    and \
                        isinstance(robot0_gripper_qpos, np.ndarray)  and \
                        isinstance(ee_state, np.ndarray) and \
                        isinstance(action, np.ndarray)
                    assert agentview_image.shape == image_shape == robot0_eye_in_hand_image.shape
                    # print(ee_state.shape)
                    # print(action.shape)
                    task = file.split('/2024')[0].split('/')[-1]
                    if task in pnp_to_pickplace:
                        task = pnp_to_pickplace[task]
                    assert task in task_to_description
                    
                    dataset.add_frame(
                        {
                            "image": agentview_image, 
                            "wrist_image": robot0_eye_in_hand_image,
                            "state": ee_state.astype(np.float32),
                            "actions": action.astype(np.float32), 
                            "task": task_to_description[task]
                        }
                    )                
                
                dataset.save_episode()
                print('STATS---> num_episodes:', dataset.num_episodes, 'total frames:', len(dataset))
                # break
            
        print('-'*20)

elif  'robomimic' in args.repo:
    task_instruction = {
        'lift': 'pick up the object on the table and hold it' ,
        'can': 'pick up the coke can and place it on the correct place',
        'square': 'pick a square nut and place it on a rod',
        'tool_hang': 'assemble a frame consisting of a base piece and hook piece by inserting the hook into the base, and hang a wrench on the hook'
        } 
    for task in ['tool_hang', 'can',  'square', 'lift']:
        for human in ['ph']:
            file = f'/mnt/disk1t/robomimic_dataset/robomimic/v1.5/{task}/{human}/demo_v15_image.hdf5'
            assert os.path.exists(file), f"File not found: {file}"
            with h5py.File(file, "r") as f:    
                # data:
                #    demo_0
                #    demo_1
                #    demo_2
                #    ....     
                print('hdf5 file===>',task,  human )   
                print(f['data'].keys())
                print('total trajectories:', len(f['data']))
                assert len(f['data']) > 100

                for traj_id, traj in f['data'].items():

                    # demo_99 ===> <KeysViewHDF5 ['actions', 'controller_info', 'interventions', 'policy_acting', 'states', 'user_acting', 'user_info']>

                    print(traj_id, '===>', 'total steps:', traj['actions'].shape[0])
                    print('traj obs==>', traj['obs'].keys())
                    # for k, v in traj.items():
                    #     print('traj.keys==>', k, np.array(v).shape)
                    #     if k == 'obs':
                    #         for ob, obv in v.items():
                    #             print('\t', ob, obv.shape )
                    
                    assert traj['actions'].shape[0] == traj['states'].shape[0] == traj['rewards'].shape[0]
                    
                    for step_ix in range(traj['actions'].shape[0]):
                        
                        # frontview_image = traj['obs']['frontview_image'][()][step_ix]
                        agentview_image = traj['obs']['agentview_image'][()][step_ix]
                        robot0_eye_in_hand_image = traj['obs']['robot0_eye_in_hand_image'][()][step_ix]
                        
                        robot0_eef_pos = traj['obs']['robot0_eef_pos'][step_ix]
                        robot0_eef_quat = traj['obs']['robot0_eef_quat'][step_ix]
                        robot0_gripper_qpos = traj['obs']['robot0_gripper_qpos'][step_ix]
                        
                        ee_state = compute_ee_state(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos) 

                        action = traj['actions'][()][step_ix]
                        
                        assert \
                            isinstance(agentview_image, np.ndarray) and \
                            isinstance(robot0_eye_in_hand_image, np.ndarray) and \
                            isinstance(robot0_eef_pos, np.ndarray)    and \
                            isinstance(robot0_eef_quat, np.ndarray)    and \
                            isinstance(robot0_gripper_qpos, np.ndarray)  and \
                            isinstance(ee_state, np.ndarray) and \
                            isinstance(action, np.ndarray)
                        assert agentview_image.shape == image_shape == robot0_eye_in_hand_image.shape
                        # print(ee_state.shape)
                        # print(action.shape)
                    
                        dataset.add_frame(
                            {
                                "image": agentview_image, 
                                "wrist_image": robot0_eye_in_hand_image,
                                "state": ee_state.astype(np.float32),
                                "actions": action.astype(np.float32), 
                                "task": task_instruction[task]
                            }
                        )                
        
                    dataset.save_episode()
                    print('STATS---> num_episodes:', dataset.num_episodes, 'total frames:', len(dataset))
                    print()



elif  'mimicgen' in args.repo:
    files = glob.glob("/mnt/disk1t/mimicgen_datasets/source/*.hdf5")
    random.shuffle(files)
    print('files cnt:', len(files))
    for file in files:
        print(file)
        if 'coffee' in file:
            task_prompt = 'make coffee'
        elif 'hammer_cleanup' in file:
            task_prompt = 'clean up the hammer'
        elif 'mug_cleanup' in file:
            task_prompt = 'clean up the mug'
        elif 'pick_place' in file:
            task_prompt = 'pick the object and place it in the right place'
        elif 'stack' in file:
            task_prompt = 'stack the cubes on the table'
        elif 'threading' in file:
            task_prompt = 'do threading'
        elif 'kitchen' in file:
            task_prompt = 'do cooking in the kitchen'
        elif 'nut_assembly' in file:
            task_prompt = 'assemble the nut'
        elif 'square' in file:
            task_prompt = 'square ring on peg'
        elif 'three_piece_assembly' in file:
            task_prompt = 'assemble the three pieces'
        else:
            raise ValueError(f"file name:{file}")
        
        
        with h5py.File(file, "r") as f:
            print(f['data'].keys())
            print('total trajectories:', len(f['data']))

            for traj_id, traj in f['data'].items():
                print(traj_id, '===>', 'total steps:', traj['actions'].shape[0])
                assert traj['actions'].shape[0] == traj['states'].shape[0] == traj['rewards'].shape[0]
                # ['actions', 'dones', 'obs', 'rewards', 'states']
                
                for step_ix in range(traj['actions'].shape[0]):
                    # print(traj['obs'].keys())
                    # ['agentview_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_vel_ang', 'robot0_eef_vel_lin', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']
                    
                    agentview_image = traj['obs']['agentview_image'][()][step_ix]
                    robot0_eye_in_hand_image = traj['obs']['robot0_eye_in_hand_image'][()][step_ix]
                    
                    robot0_eef_pos = traj['obs']['robot0_eef_pos'][step_ix]
                    robot0_eef_quat = traj['obs']['robot0_eef_quat'][step_ix]
                    robot0_gripper_qpos = traj['obs']['robot0_gripper_qpos'][step_ix]
                    
                    ee_state = compute_ee_state(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos) 

                    action = traj['actions'][()][step_ix]
                    
                    assert \
                        isinstance(agentview_image, np.ndarray) and \
                        isinstance(robot0_eye_in_hand_image, np.ndarray) and \
                        isinstance(robot0_eef_pos, np.ndarray)    and \
                        isinstance(robot0_eef_quat, np.ndarray)    and \
                        isinstance(robot0_gripper_qpos, np.ndarray)  and \
                        isinstance(ee_state, np.ndarray) and \
                        isinstance(action, np.ndarray)
                    
                    assert ee_state.shape == (8,)
                    assert agentview_image.shape == image_shape == robot0_eye_in_hand_image.shape
                    assert action.shape == (7,)
                
                    dataset.add_frame(
                        {
                            "image": agentview_image, 
                            "wrist_image": robot0_eye_in_hand_image,
                            "state": ee_state.astype(np.float32),
                            "actions": action.astype(np.float32), 
                            "task": task_prompt
                        }
                    )                
    
                dataset.save_episode()
                print('STATS---> num_episodes:', dataset.num_episodes, 'total frames:', len(dataset))
                print()                
                
            print()
            # os._exit(0)
        print('-'*10)



elif 'genesis' in args.repo:
    assert 'delta' in args.repo or 'abs' in args.repo
    
    files  = glob.glob("/home/yanan/robotics/trajectories_vr/*.npz")
    random.shuffle(files)
    for file in files:
        traj = np.load(file, allow_pickle=True)
        print(file, '===>', len(traj["data"].tolist()))
        # dict_keys(['frontview_image', 'agentview_image', 'wrist_image', 'state', 'actions', 'task'])
        # if 'delta' in args.repo and 'actions_delta_ee' not in traj["data"].tolist()[1]:
        #     continue

        idle_cnt = 0
        for ii in range(len(traj["data"].tolist())):
            step = traj["data"].tolist()[ii]
            assert 'image' in step.keys() and 'wrist_image' in step.keys() and 'state' in step.keys() and 'actions_delta_ee' in step.keys()

            # filter out idle actions
            if ii == 0 :
                prior_action_abs = step['actions']
            else:
                if (np.max(np.abs(prior_action_abs - step['actions'])) < 0.001 or np.max(np.abs(step['actions_delta_ee'])) < 0.001)  and prior_action_abs[-1] != 1 : 
                    idle_cnt += 1
                    continue
                prior_action_abs = step['actions']
                
                
            if 'delta' in args.repo:
                step['actions'] = step.pop('actions_delta_ee') 
            elif 'abs' in args.repo:
                if 'actions_delta_ee' in step.keys():
                    del step['actions_delta_ee']
                if 'actions_abs_ee' in step.keys():
                    step['actions'] = step.pop('actions_abs_ee')
                    
            
            assert 'actions' in step.keys()

            # gripper is open status
            if step['actions'][-1] == 0:
                step['actions'][-1] = -1 
        
            assert step['actions'][-1] in [-1, 1]
            
            dataset.add_frame(step)
            
            # image = Image.fromarray(step['image'], mode="RGB")
            # image.show()
            # time.sleep(3)

            # image = Image.fromarray(step['wrist_image'], mode="RGB")
            # image.show()
            # time.sleep(3)
        
        print('idle filter out:', round(idle_cnt / len(traj["data"].tolist()) ,4) )
        dataset.save_episode()    
        print('STATS---> num_episodes:', dataset.num_episodes, 'total frames:', len(dataset))

print(dataset)
if lerobot.__version__.startswith("0.4."):
    dataset.finalize()


if args.push:
    assert dataset.repo_id
    print('begin to push to hub:', dataset.repo_id)
    dataset.push_to_hub(
        tags=["libero", "panda", "franka"],
        private=False,
        push_videos=True,
        license="apache-2.0",
    )


