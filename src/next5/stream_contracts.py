"""Instrumented one-shot streams used only by the independent API auditor."""


class Online:
    def __init__(self,points):self.points=points;self.consumed=0;self.iterated=False
    def __iter__(self):
        if self.iterated:raise AssertionError('Online iterator traversed twice')
        self.iterated=True;return self
    def __next__(self):
        if self.consumed==len(self.points):raise StopIteration
        point=self.points[self.consumed];self.consumed+=1;return point
    def __len__(self):raise AssertionError('Future online length inspected')
    def __length_hint__(self):raise AssertionError('Future online length hint inspected')
    def __array__(self,*args,**kwargs):raise AssertionError('Future online suffix materialized')


class Datasets:
    def __init__(self,records):self.records=iter(records);self.current=None;self.accessed=0;self.sid=None
    def __iter__(self):return self
    def __next__(self):
        if self.current is not None and self.current.consumed!=len(self.current.points):raise AssertionError('Next dataset fetched before current timeline exhausted')
        sid,historical,points=next(self.records);self.sid=sid;self.current=Online(points);self.accessed+=1
        return historical,self.current
    def __len__(self):raise AssertionError('Dataset iterator length inspected')
    def __length_hint__(self):raise AssertionError('Dataset iterator length hint inspected')
