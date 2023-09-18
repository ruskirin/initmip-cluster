from pathlib import Path
import ipywidgets as pwidg
from IPython.display import display
from collections import namedtuple
import regex as re
from typing import Callable

from ghub_utils import files as gfiles
from ghub_utils import widgets as gwidg
import files as mfiles


class FeatureSelector(pwidg.VBox):
    """
    TODO 8/11: Docs here
    """
    OPTS_START = ['Fields', 'Models']

    def __init__(self, dir_models: Path, **kwargs):
        """
        :param dir_models:
        :param kwargs:
        :return:
        """
        # --- WIDGETS ---
        self.selection = None

        option_start = gwidg.OptionToggle(
            label='Select:', options=self.OPTS_START
        )
        box_selectors = FileSelectors(dir_models)
        btn_submit = pwidg.Button(description='Select')
        output = pwidg.Output()

        # --- ACTIONS ---
        # disable 'Models' option for now
        option_start.disable(self.OPTS_START[1])
        # hide selectors
        box_selectors.hide()
        # hide button
        btn_submit.disabled = True
        btn_submit.layout.display = 'none'

        # TODO 8/20: atm, select_fields/models() only show the selectors; planned
        #   to allow selecting in opposite direction  too -- models first
        #   -> exp -> fields. But it might be unnecessary
        def select_fields(btn):
            box_selectors.show()
            btn_submit.layout.display = 'block'

        def select_models(btn):
            box_selectors.show()
            btn_submit.layout.display = 'block'

        option_start.on_click(
            {self.OPTS_START[0]: select_fields,
             self.OPTS_START[1]: select_models}
        )

        def enable_submit(change):
            btn_submit.disabled = False

        box_selectors.m_observe(enable_submit)

        @output.capture(clear_output=True, wait=True)
        def submit(btn):
            self.selection = box_selectors.selection
            with output:
                print(f'{len(self.selection.files)} files loaded:'
                      f'\n- Experiment: {self.selection.exp}'
                      f'\n- Models: {self.selection.models}'
                      f'\n- Fields: {self.selection.fields}')

        btn_submit.on_click(submit)

        super().__init__(
            children=(option_start, box_selectors, btn_submit, output),
            **kwargs
        )


class FileSelectors(pwidg.HBox):
    """
    HBox of selectors for models, experiments, and fields
    """
    LAYOUT_BOX = pwidg.Layout(
        width='auto',
        border='solid'
    )

    def __init__(self, dir_models: Path, groups: dict = None, **kwargs):
        """
        :param dir_models: 
        :param groups: (optional) dictionary of 
          {field: tuple of component fields} pairs; component fields are those
          fields that always go together to form another field 
        :param kwargs: 
        """
        # recursively get all .nc files starting from @dir_models_a
        self._files_all = list(dir_models.rglob('*.nc'))
        self._features_all = mfiles.union_netcdf_params(self._files_all)

        Selection = namedtuple('Selection', 'files exp fields models')
        self._selection = None
        # Callbacks for @selection attribute
        self._callbacks = []

        radio_exps = pwidg.RadioButtons(
            options=sorted(self._features_all.exps),
            description='Experiment:'
        )
        sel_fields = MultiSelector(
            select_label='Fields:',
            options=sorted(self._features_all.fields),
            disabled=True
        )
        sel_models = MultiSelector(
            select_label='Models:',
            options=sorted(self._features_all.models),
            disabled=True
        )

        def select_exp(change):
            """Experiment selected"""
            exp = change['new']
            # clear selected fields and models
            sel_fields.value = []
            sel_models.value = []

            if (exp is not None) and len(exp) > 0:
                sel_fields.enable()

                filter = mfiles.filter_paths_terms(
                    self._files_all, [exp,], mfiles.FileParams.EXP
                )

                params = mfiles.union_netcdf_params(filter)
                sel_fields.options = sorted(params.fields)
                sel_models.options = sorted(params.models)

        radio_exps.observe(select_exp, names='value')

        def select_fields(change):
            """Fields selected"""
            exp = radio_exps.value
            fields = change['new']

            sel_models.value = []

            if (fields is not None) and len(fields) > 0:
                sel_models.enable()

                filter = mfiles.filter_paths_terms(
                    self._files_all, [exp, ], mfiles.FileParams.EXP
                )
                filter = mfiles.filter_paths_terms(
                    filter, fields, mfiles.FileParams.FIELD
                )
                models = mfiles.intersect_netcdf_model_params(filter)
                sel_models.options = sorted(models)

        sel_fields.mobserve(select_fields, names='value')

        def select_models(change):
            """
            Models selected; update @self._selection attribute and notify any
              observers
            """
            exp = radio_exps.value
            fields = sel_fields.value
            models = change['new']

            if (models is not None) and len(models) > 0:
                # Filter .nc files according to selections
                filter = mfiles.filter_paths_terms(
                    self._files_all, [exp,], mfiles.FileParams.EXP
                )
                filter = mfiles.filter_paths_terms(
                    filter, fields, mfiles.FileParams.FIELD
                )
                filter = mfiles.filter_paths_terms(
                    filter, models, mfiles.FileParams.MODEL
                )

                self.selection = Selection(filter, exp, fields, models)

        sel_models.mobserve(select_models, names='value')

        children = (radio_exps, sel_fields, sel_models)
        super().__init__(
            children=children,
            layout=self.LAYOUT_BOX,
            **kwargs
        )

    @property
    def selection(self):
        return self._selection
    @selection.setter
    def selection(self, val: namedtuple):
        self._selection = val
        # notify registered observers
        for cb in self._callbacks:
            cb(val)
    def m_observe(self, callback):
        """Register observer @callback"""
        self._callbacks.append(callback)

    def hide(self):
        self.layout.display = 'none'

    def show(self):
        self.layout.display = 'block'


class MultiSelector(pwidg.VBox):
    """
    Selector widget with a "select all" button
    """

    def __init__(self,
                 select_label: str,
                 options: list,
                 **kwargs):
        """
        :param select_label:
        :param options:
        :param multiple:
        :param kwargs:
        """
        # --- PROPERTIES ---
        label_sel = pwidg.Label(value=select_label)
        selector = pwidg.SelectMultiple(
            options=options, layout=pwidg.Layout(display='flex')
        )
        btn_sel_all = pwidg.Button(description=f'All ({len(selector.options)})')
        box_label = pwidg.HBox(children=(label_sel, btn_sel_all))

        # --- ACTIONS ---
        def select_all(b):
            selector.value = selector.options
        btn_sel_all.on_click(select_all)

        output = pwidg.Output() # shows selected options

        super().__init__(
            children=(box_label, selector, output),
            **kwargs
        )

        if kwargs.get('disabled'):
            self.disable()

    @property
    def options(self):
        selector = self.children[1]
        return selector.options
    @options.setter
    def options(self, val):
        selector = self.children[1]
        selector.options = val

        btn_all = self.children[0].children[1]
        btn_all.description = f'All ({len(val)})'

    @property
    def value(self):
        selector = self.children[1]
        return selector.value
    @value.setter
    def value(self, val):
        selector = self.children[1]
        selector.value = val

    def mobserve(self, func: Callable[[dict], None], **kwargs):
        """Set an observer on the selector"""
        sel: pwidg.Select = self.children[1]
        sel.observe(func, **kwargs)

    def enable(self):
        for child in self.children:
            if isinstance(child, pwidg.Box):
                for c in child.children:
                    c.disabled = False

            child.disabled = False

    def disable(self):
        for child in self.children:
            if isinstance(child, pwidg.Box):
                for c in child.children:
                    c.disabled = True

            child.disabled = True


if __name__ == '__main__':
    models_dir = gfiles.DIR_SAMPLE_DATA / 'models'
    dirs = [models_dir,]

    f = FileSelectors(models_dir)
    exps = f.children[0]
    fields = f.children[1]
    models = f.children[2]

    exps.value = exps.options[0]
    fields.value = fields.options[:5]
    models.value = models.options[:]

    display(f.selection)