
# Bird Songs Dataset

This dataset contains audio recordings of various bird species sourced from the Xeno-canto platform.

## Source of Data
The recordings were obtained from the Xeno-canto API, which provides access to a large collection of bird sounds from around the world.

## How It Was Created
The dataset was created by querying the Xeno-canto API for specific bird genera and filtering the results to include only those recordings that have a suitable license for commercial use. Each bird genus was queried to retrieve five recordings.

## License Type
The recordings included in this dataset are filtered to ensure they have licenses that allow for commercial use. Specifically, only recordings with licenses that do not include "nc" (non-commercial) were included.

## Dataset Structure
The dataset contains the following fields:
- `id`: The unique identifier for the recording.
- `name`: The English name of the bird species.
- `url`: The URL to download the audio file.
- `license`: The license type of the recording.
