import logging
import os
import genesis as gs

logging.basicConfig(level=logging.DEBUG)

def run_genesis_simulation(simulation_type: str, parameters: dict, show_viewer: bool) -> dict:
    """
    Run a Genesis simulation based on the provided type and parameters.

    Args:
        simulation_type (str): e.g., 'rigid_body' or 'fluid'.
        parameters (dict): Simulation configuration.
            - 'file' (string): The URDF/MJCF filename (e.g. 'cf2x.urdf').
            - 'assets_dir' (string, optional): The folder containing the file.
            - 'scale', 'pos', 'euler', etc.
            - 'simulation_steps' (int): How many timesteps to run.
        show_viewer (bool): Whether to open a viewer (often False on EC2).

    Returns:
        dict: A status dictionary with "status" and "message".
    """
    try:
        logging.debug("=== [Genesis] Entering run_genesis_simulation ===")

        # 1) Initialize Genesis once if not already done
        if not hasattr(gs, "_initialized") or not getattr(gs, "_initialized", False):
            if hasattr(gs, "init"):
                logging.debug("Calling gs.init(backend=gs.cpu)")
                gs.init(backend=gs.cpu)
                gs._initialized = True
                logging.debug("Genesis backend initialized.")
            else:
                logging.warning("Genesis init not found. Initialization skipped.")
        else:
            logging.debug("Genesis already initialized. Skipping re-initialization.")

        # 2) Log incoming parameters
        logging.debug(f"Simulation Type: {simulation_type}")
        logging.debug(f"Parameters: {parameters}")
        logging.debug(f"Show Viewer: {show_viewer}")

        # 3) Create Sim and Viewer options (updated for new Genesis version)
        sim_opts = {
            "dt": parameters.get("dt", 0.01),
            "substeps": parameters.get("substeps", 1),
            "gravity": parameters.get("gravity", (0, 0, -9.8))
        }

        viewer_opts = {
            "camera_pos": parameters.get("camera_pos", (3.5, 0.0, 2.5)),
            "camera_lookat": parameters.get("camera_lookat", (0.0, 0.0, 0.5)),
            "camera_fov": parameters.get("camera_fov", 40),
            "max_FPS": parameters.get("fps", 60)
        }

        # 4) Create the Genesis Scene using the new API if available.
        try:
            scene = gs.Scene()
        except Exception as e:
            logging.error(f"Failed to create scene using gs.Scene: {e}")
            scene = None

        if scene is None:
            return {
                "status": "error",
                "message": "Scene creation is not supported in this version of Genesis."
            }

        # 5) Build the full file path if assets_dir is provided
        file_name = parameters["file"]  # required
        assets_dir = parameters.get("assets_dir", None)
        if assets_dir:
            logging.debug(f"Using assets_dir: {assets_dir}")
            full_file_path = os.path.join(assets_dir, file_name)
        else:
            full_file_path = file_name

        # 6) Determine which morph/entity is needed
        if simulation_type == "rigid_body":
            extension = os.path.splitext(full_file_path)[1].lower()
            if extension == ".urdf":
                logging.debug("Using gs.morphs.URDF for rigid_body simulation.")
                morph = gs.morphs.URDF(
                    file=full_file_path,
                    collision=True,
                    scale=parameters.get("scale", 1.0),
                    pos=parameters.get("pos", (0.0, 0.0, 0.0)),
                    euler=parameters.get("euler", (0.0, 0.0, 0.0)),
                )
            elif extension == ".xml":
                logging.debug("Using gs.morphs.MJCF for rigid_body simulation.")
                morph = gs.morphs.MJCF(
                    file=full_file_path,
                    collision=True,
                    scale=parameters.get("scale", 1.0),
                    pos=parameters.get("pos", (0.0, 0.0, 0.0)),
                    euler=parameters.get("euler", (0.0, 0.0, 0.0)),
                )
            else:
                msg = f"Unsupported file extension {extension} for 'rigid_body' simulation."
                logging.error(msg)
                raise ValueError(msg)

        elif simulation_type == "fluid":
            logging.debug("Using gs.morphs.MJCF for fluid simulation.")
            morph = gs.morphs.MJCF(
                file=full_file_path,
                collision=False,
                scale=parameters.get("scale", 1.0),
                pos=parameters.get("pos", (0.0, 0.0, 0.0)),
                euler=parameters.get("euler", (0.0, 0.0, 0.0)),
            )
        else:
            msg = f"Unsupported simulation type: {simulation_type}"
            logging.error(msg)
            raise ValueError(msg)

        # 7) Add the entity to the Scene
        logging.debug("Adding entity to the scene...")
        entity = scene.add_entity(morph)
        logging.debug(f"Entity added: {entity}")

        # 8) Build the scene
        logging.debug("Building the scene (compile_kernels set to False by default).")
        scene.build(compile_kernels=parameters.get("compile_kernels", False))
        logging.debug("Scene build complete.")

        # 9) Simulation loop
        steps = parameters.get("simulation_steps", 100)
        logging.debug(f"Starting simulation loop for {steps} steps...")
        for step in range(steps):
            logging.debug(f"--- Simulation step: {step} ---")
            scene.step()

        # 10) Return success response
        logging.debug("All steps complete. Returning success response...")

        return {
            "status": "success",
            "message": f"{simulation_type} simulation completed successfully."
        }

    except Exception as e:
        logging.error(f"Simulation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}