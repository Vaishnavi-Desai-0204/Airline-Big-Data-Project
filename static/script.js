function createBarChart(data) {
    var margin = {top: 20, right: 20, bottom: 30, left: 40};
    var width = 600 - margin.left - margin.right;
    var height = 400 - margin.top - margin.bottom;
  
    var svg = d3.select('#chart')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
  
    var x = d3.scaleBand()
              .range([0, width])
              .padding(0.1)
              .domain(data.map(function(d) { return d['Source Airport'] + ' to ' + d['Destination Airport']; }));
  
    var y = d3.scaleLinear()
              .range([height, 0])
              .domain([0, d3.max(data, function(d) { return d['Num flights']; })]);
  
    svg.append('g')
       .attr('class', 'x axis')
       .attr('transform', 'translate(0,' + height + ')')
       .call(d3.axisBottom(x));
  
    svg.append('g')
       .attr('class', 'y axis')
       .call(d3.axisLeft(y).ticks(5));
  
    svg.selectAll('.bar')
       .data(data)
       .enter().append('rect')
       .attr('class', 'bar')
       .attr('x', function(d) { return x(d['Source Airport'] + ' to ' + d['Destination Airport']); })
       .attr('y', function(d) { return y(d['Num flights']); })
       .attr('width', x.bandwidth())
       .attr('height', function(d) { return height - y(d['Num flights']); });
  }
  
  