# Google Earth Engine Setup

This guide explains how to securely initialize Google Earth Engine whenever
you reopen the SAR Lab 7 project. Do not save the real Google Cloud project ID
in this repository or notebook.

## One-Time Authentication

Activate the lab environment:

```bash
conda activate sar-lab-7
```

Authenticate Earth Engine:

```bash
earthengine authenticate
```

Complete the browser authorization using the Google account that has Earth
Engine access.

Authentication credentials are stored in the user profile, outside this
repository. Repeat authentication only if the credentials expire or the
account changes:

```bash
earthengine authenticate --force
```

## Recommended: Set the Project ID Per Terminal Session

Open the PyCharm Terminal and activate the environment:

```bash
conda activate sar-lab-7
cd /Users/ernestmolczan/PyCharmProjects/eolabs-clone/SAR_Lab_7
```

Enter the project ID without displaying it:

```bash
read -s "EARTH_ENGINE_PROJECT?GEE project ID: "
export EARTH_ENGINE_PROJECT
echo
```

Confirm that the variable exists without printing its value:

```bash
test -n "$EARTH_ENGINE_PROJECT" && echo "Earth Engine project ID is set"
```

Verify Earth Engine access:

```bash
python verify_gee_setup.py
```

Expected output:

```text
Earth Engine verification succeeded.
Sentinel-1 metadata probe count: ...
Project ID was loaded from the environment and was not displayed.
```

Start Jupyter from the same terminal so it inherits the variable:

```bash
jupyter lab
```

Open `SAR_Lab_7_analysis.ipynb` and select the `Python (SAR Lab 7)` kernel.

When finished, close Jupyter and remove the variable:

```bash
unset EARTH_ENGINE_PROJECT
```

## PyCharm Run Configuration

Processes started with PyCharm's Run button do not inherit variables exported
in an unrelated terminal.

To run `verify_gee_setup.py` directly from PyCharm:

1. Open **Run > Edit Configurations**.
2. Select or create a Python configuration for `verify_gee_setup.py`.
3. Set the interpreter to:

   ```text
   ~/miniforge3/envs/sar-lab-7/bin/python
   ```

4. Open **Environment variables**.
5. Add:

   ```text
   EARTH_ENGINE_PROJECT=your-project-id
   ```

6. Apply the configuration and run it.

This is less private than the terminal method because PyCharm may persist the
value in local `.idea/` configuration files. The `.idea/` directory should
remain untracked and must not be shared.

## Every Time the Project Is Reopened

Use this short sequence:

```bash
conda activate sar-lab-7
cd /Users/ernestmolczan/PyCharmProjects/eolabs-clone/SAR_Lab_7

read -s "EARTH_ENGINE_PROJECT?GEE project ID: "
export EARTH_ENGINE_PROJECT
echo

python verify_gee_setup.py
jupyter lab
```

After the work session:

```bash
unset EARTH_ENGINE_PROJECT
```

## Troubleshooting

### Authentication is missing

```bash
earthengine authenticate
```

### Authentication is expired or uses the wrong account

```bash
earthengine authenticate --force
```

### `EARTH_ENGINE_PROJECT` is not set

The variable exists only in the terminal session where it was exported. Set it
again and start Python or Jupyter from that same terminal.

### PyCharm cannot see the variable

Either run the script from the terminal where the variable was exported or add
it to the script's PyCharm Run Configuration.

### Project or permission error

Verify that:

- The project ID is correct.
- The Google Cloud project is registered for Earth Engine.
- The authenticated Google account can use that project.
- The Earth Engine API is enabled for the project.

## Security Rules

- Never put the real project ID directly into the notebook.
- Never commit credentials, tokens, `.env` files, or PyCharm secret settings.
- Never print the project ID in notebook output or screenshots.
- Do not share `~/.config/earthengine/credentials`.
- Use `unset EARTH_ENGINE_PROJECT` after each session.
