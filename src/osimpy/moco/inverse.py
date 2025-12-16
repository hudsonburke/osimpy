import opensim as osim


# Adapted from OpenSim's exampleMocoInverse.py
def solveMocoInverse(
    model_file: str,
    external_loads_file: str,
    coordinates_file: str,
    solution_path: str | None = None,
    initial_time: float = 0.0,
    final_time: float = -1.0,
    mesh_interval: float = 0.02,
    active_fiber_force_scale: float = 1.5,
    muscle_path_set_file: str | None = None,
    generate_report: bool = True,
    bilateral: bool = True,
):
    # Construct the MocoInverse tool.
    inverse = osim.MocoInverse()

    # Construct a ModelProcessor and set it on the tool. The default
    # muscles in the model are replaced with optimization-friendly
    # DeGrooteFregly2016Muscles, and adjustments are made to the default muscle
    # parameters.
    modelProcessor = osim.ModelProcessor(model_file)
    modelProcessor.append(osim.ModOpAddExternalLoads(external_loads_file))
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    # Only valid for DeGrooteFregly2016Muscles.
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    # Only valid for DeGrooteFregly2016Muscles.
    modelProcessor.append(
        osim.ModOpScaleActiveFiberForceCurveWidthDGF(active_fiber_force_scale)
    )
    # Use a function-based representation for the muscle paths. This is
    # recommended to speed up convergence, but if you would like to use
    # the original GeometryPath muscle wrapping instead, simply comment out
    # this line. To learn how to create a set of function-based paths for
    # your model, see the example 'examplePolynomialPathFitter.py'.
    if muscle_path_set_file:
        modelProcessor.append(
            osim.ModOpReplacePathsWithFunctionBasedPaths(muscle_path_set_file)
        )
    modelProcessor.append(osim.ModOpAddReserves(1.0))
    inverse.setModel(modelProcessor)

    # Construct a TableProcessor of the coordinate data and pass it to the
    # inverse tool. TableProcessors can be used in the same way as
    # ModelProcessors by appending TableOperators to modify the base table.
    # A TableProcessor with no operators, as we have here, simply returns the
    # base table.
    inverse.setKinematics(osim.TableProcessor(coordinates_file))

    # Initial time, final time, and mesh interval.
    inverse.set_initial_time(initial_time)
    inverse.set_final_time(final_time)
    inverse.set_mesh_interval(mesh_interval)

    # By default, Moco gives an error if the kinematics contains extra columns.
    # Here, we tell Moco to allow (and ignore) those extra columns.
    inverse.set_kinematics_allow_extra_columns(True)

    # Solve the problem and write the solution to a Storage file.
    solution = inverse.solve()
    solution_path = solution_path or coordinates_file.replace(
        ".sto", "_MocoInverse_solution.sto"
    )
    solution.getMocoSolution().write(solution_path)

    if generate_report:
        # Generate a PDF with plots for the solution trajectory.
        model = modelProcessor.process()
        report = osim.report.Report(model, solution_path, bilateral=bilateral)
        # The PDF is saved to the working directory.
        report.generate()
