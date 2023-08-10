from pathlib import Path
import ipywidgets as pwidg
from IPython.display import display
from enum import Enum

from ghub_utils import files


class FileSelector(pwidg.VBox):
    """
    Selector widget that displays the desired contents of a directory and allows
      for their selection
    """
    class FileType(Enum):
        """acceptable file types in FileSelectors"""
        DIR = 'dir'
        ANY = None
        NC = 'nc'

    def __init__(self,
                 select_label: str,
                 directory: Path,
                 multiple: bool,
                 file_type: FileType,
                 **kwargs):
        self.selection = []
        options: dict = self._create_options(directory, file_type)

        label_sel = pwidg.Label(value=select_label)

        if multiple:
            selector = pwidg.SelectMultiple(options=list(options.keys()))
        else:
            selector = pwidg.Select(options=list(options.keys()))

        btn_submit = pwidg.Button(description='Select')
        btn_sel_all = pwidg.Button(description='All')
        btn_sel_none = pwidg.Button(description='Clear')
        box_btn_sel = pwidg.HBox(
            children=(btn_submit, btn_sel_all, btn_sel_none)
        )

        def select_all(b): selector.value = selector.options
        btn_sel_all.on_click(select_all)

        def select_clear(b): selector.value = ()
        btn_sel_none.on_click(select_clear)

        output = pwidg.Output() # shows selected options

        @output.capture(clear_output=True, wait=True)
        def select(b):
            self.selection = {n: options[n] for n in list(selector.value)}
            print(f'Selected: {list(selector.value)}')
        btn_submit.on_click(select)

        super().__init__(
            children=(label_sel, selector, box_btn_sel, output),
            **kwargs
        )

    def _create_options(self, directory: Path, file_type: FileType) -> dict:
        """Create a list of options to display in the selector"""
        raise NotImplementedError(
            f'FileSelector children must overwrite _create_options()'
        )


class ModelSelector(FileSelector):
    """List out models from the 'models' directory"""
    def __init__(self, directory: Path, **kwargs):
        super().__init__(
            select_label='Models:',
            directory=directory,
            multiple=True,
            file_type=FileSelector.FileType.DIR,
            **kwargs
        )

    def _create_options(
            self, directory: Path, file_type: FileSelector.FileType
    ) -> dict:
        options = {p.name: p for p in sorted(directory.iterdir()) if p.is_dir()}
        return options


# if __name__ == '__main__':
#     print(FileSelector.FileType.DIR.value)